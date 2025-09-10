from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import spacy
import dateparser
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Run with: uvicorn backend.ai_llm_backend:app --reload --port 8000
# RUN FROM PROJECT ROOT


#need to handle available from and also ignore events that arent well formed so it can post anyway
# -----------------------------
# pydanitc Models
# -----------------------------
class AssignmentItem(BaseModel):
    assignment: str
    due_date: Optional[str] = None
    time: Optional[str] = None

class Assignments(BaseModel):
    assignments: List[AssignmentItem]

class ExtractRequest(BaseModel):
    text: str

# -----------------------------
# FastAPI App => React
# -----------------------------
app = FastAPI(title="Assignment Extractor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3000/deadline_tracker", 
        "https://cbc612407873.ngrok-free.app" #ngrok 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load models
# -----------------------------
nlp_token_extractor = spacy.load("en_core_web_trf")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

# -----------------------------
# Utilities (from your script)
# -----------------------------
def normalize_time(text: str):
    dt = dateparser.parse(text)
    if not dt:
        return None, None
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M") if (dt.hour or dt.minute) else None
    return date_str, time_str

def remove_ignore_lines(text: str) -> str:
    #remove everything inside <IGNORE>...</IGNORE>
    cleaned_text = re.sub(r"<IGNORE>.*?</IGNORE>", "", text, flags=re.DOTALL | re.IGNORECASE)
    #remove blank lines
    cleaned_text = "\n".join([line for line in cleaned_text.split("\n") if line.strip()])
    return cleaned_text

def clean_due_tags(block_text: str):
    # search for the 'Due' keyword
    match = re.search(r'\bDue\b', block_text, flags=re.IGNORECASE)
    if not match:
        return block_text 
    before_due = block_text[:match.end()]
    after_due = block_text[match.end():]
    #find all <DUE> tags in the text after 'Due'
    due_tags = re.findall(r'<DUE>.*?</DUE>', after_due)
    if not due_tags:
        return block_text
    first_due = due_tags[0]
    #remove all <DUE> tags in the remainder
    after_due_cleaned = re.sub(r'<DUE>.*?</DUE>', '', after_due)
    # readd only the first <DUE>
    after_due_cleaned = first_due + after_due_cleaned
    return before_due + after_due_cleaned


def clean_all_assignment_blocks(marked_text: str):
    def clean_block(match):
        block = match.group(0)
        return clean_due_tags(block)
    cleaned_text = re.sub(r'<ASSIGNMENT>.*?</ASSIGNMENT>', clean_block, marked_text, flags=re.DOTALL)
    return cleaned_text

def mark_tags(text: str):
    """
    Mark assignments text with NER tags:
    - <ASSIGNMENT>...</ASSIGNMENT> around each assignment block
    - <ASSIGNMENT_NAME>...</ASSIGNMENT_NAME> for the assignment title
    - <IGNORE>...</IGNORE> for 'Not available' lines and grading lines
    - <DUE><DATE>...</DATE> <TIME>...</TIME></DUE> for due dates/times
    """
    def inside_tag(text, start, end, tag):
        pattern = fr"<{tag}>(.*?)</{tag}>"
        for m in re.finditer(pattern, text, flags=re.DOTALL):
            if m.start(1) <= start and end <= m.end(1):
                return True
        return False

   #blocks = re.split(r'(?=Assignment\b)', text)
    blocks = re.split(r'(?=(Assignment|Quiz)\b)', text)
    #blocks = re.split(r'(?:Assignment|Quiz)\b', text)
    marked_blocks = []
    for block in blocks:
        print("Block:", block)
        if not block.strip():
            continue
        if block.strip() in ["Assignment", "Quiz"]:
            continue
        lines = block.split("\n")
        # Step 1: mark "Not available" and grading/points lines
        for i, line in enumerate(lines):
            if ("not available" in line.lower()) or ("points possible" in line.lower()) or ("no submission" in line.lower()) or ("available until" in line.lower()):
                lines[i] = f"<IGNORE>{line}</IGNORE>"
        #Step 1.5: remove lines that are exactly "Assignment" or "Quiz"
        for i, line in enumerate(lines):
            if re.fullmatch(r'\s*(Assignment|Quiz)\s*', line, flags=re.IGNORECASE):
                lines[i] = ""   # delete only if the line is exactly "Assignment" or "Quiz"
        # Step 2: wrap assignment name (first non-empty line not ignored)
        for i, line in enumerate(lines):
            if line.strip() and not ((line == "Assignment") or (line == "Quiz") or line.lower().startswith("due")) and "<IGNORE>" not in line:
                print("170", line)
                lines[i] = f"<ASSIGNMENT_NAME>{line.strip()}</ASSIGNMENT_NAME>"
                break
        block_text = "\n".join(lines)
        # Step 3: spaCy DATE/TIME entities
        doc = nlp_token_extractor(block_text)
        ents = list(doc.ents)
        merged = []
        skip_next = False
        for i, ent in enumerate(ents):
            if skip_next:
                skip_next = False
                continue
            # Skip entities inside <IGNORE> and <ASSIGNMENT_NAME>
            if inside_tag(block_text, ent.start_char, ent.end_char, "IGNORE") \
                or inside_tag(block_text, ent.start_char, ent.end_char, "ASSIGNMENT_NAME"):
                print("Skipping entity inside IGNORE or ASSIGNMENT_NAME:", ent.text)
            continue
            # Merge DATE + TIME
            if ent.label_ == "DATE" and i + 1 < len(ents) and ents[i+1].label_ == "TIME":
                merged.append({
                    "text": block_text[ent.start_char:ents[i+1].end_char],
                    "start": ent.start_char,
                    "end": ents[i+1].end_char
                })
                skip_next = True
            else:
                merged.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        # Step 4: replace spaCy entities and wrap in <DUE>
        marked_block = block_text
        offset = 0
        for m in merged:
            date_str, time_str = normalize_time(m["text"])
            if not date_str and not time_str:
                continue
            start = m["start"] + offset
            end = m["end"] + offset
            parts = []
            if date_str:
                parts.append(f"<DATE>{date_str}</DATE>")
            if time_str:
                parts.append(f"<TIME>{time_str}</TIME>")
            replacement = "<DUE>" + " ".join(parts) + "</DUE>"
            marked_block = marked_block[:start] + replacement + marked_block[end:]
            offset += len(replacement) - (end - start)
        # Step 5: regex safeguard for anything SpaCy missed
        def regex_replace_due(match):
            d, t = normalize_time(match.group(0))
            if not d and not t:
                return match.group(0)
            parts = []
            if d:
                parts.append(f"<DATE>{d}</DATE>")
            if t:
                parts.append(f"<TIME>{t}</TIME>")
            return "<DUE>" + " ".join(parts) + "</DUE>"
        marked_block = re.sub(
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?: at \d{1,2}:\d{2}\s?(?:am|pm))?\b',
            regex_replace_due,
            marked_block,
            flags=re.IGNORECASE
        )
        marked_blocks.append(f"<ASSIGNMENT>{marked_block}</ASSIGNMENT>")
    return "\n".join(marked_blocks)

# build prompt
# For prompt building
def createMessages(assignments_input: str):
    return [
        {
            "role": "user",
            "content": f"""
            You are an assistant that extracts assignments and their due dates and times from structured text marked with <ASSIGNMENT> tags. MAKE SURE TO GO THROUGH ALL <ASSIGNMENT>...</ASSIGNMENT> BLOCKS.  

Rules:
- Each assignment is enclosed in <ASSIGNMENT>...</ASSIGNMENT>.
- The assignment title is within <ASSIGNMENT_NAME>...</ASSIGNMENT_NAME> but ignore any <DUE>..</DUE> tags within.
- The due date is within <DUE><DATE>...</DATE></DUE>.
- The due time, if present, is within <DUE><TIME>...</TIME></DUE> or in the <TIME> tag inside <DUE>.
- Ignore any text outside <ASSIGNMENT> blocks.

Output format:
- JSON array of objects.
- Each object should have:
  - "assignment" (string) — the assignment name inside the <ASSIGNMENT_NAME>...</ASSIGNMENT_NAME> tags
  - "due_date" (string, YYYY-MM-DD format)
  - "time" (string, 24-hour format HH:MM, or null if not provided)
            
            Input: {assignments_input}"""
            
        }
    ]

def postprocess_json(generated_json: str, raw_text: str = ""):
    """
    Normalize model JSON output:
    - Ensure due_date is YYYY-MM-DD
    - Ensure time is HH:MM (24h) if present
    - Ignore dates/times from 'Not available' sections
    """
    if not generated_json.strip():
        # empty string from model
        return []
    try:
        data = json.loads(generated_json)
    except json.JSONDecodeError:
        # fallback: try to extract JSON substring from model output
        start = generated_json.find("[")
        end = generated_json.rfind("]") + 1
        if start != -1 and end != -1:
            try:
                data = json.loads(generated_json[start:end])
            except json.JSONDecodeError:
                print("Error: Could not parse JSON from model output.")
                return []
        else:
            print("Error: No JSON array found in model output.")
            return []
    for item in data:
        # Skip assignments that say "Not available"
        if "assignment" in item and "not available" in item["assignment"].lower():
            item.pop("due_date", None)
            item.pop("time", None)
            continue
        # Normalize due_date
        if "due_date" in item and item["due_date"]:
            d, t = normalize_time(item["due_date"])
            if d:
                item["due_date"] = d
            else:
                item.pop("due_date", None)
            # If time sneaks into due_date, pull it out
            if t and "time" not in item:
                item["time"] = t
        # Normalize time
        if "time" in item and item["time"]:
            _, t = normalize_time(item["time"])
            if t:
                item["time"] = t
            else:
                item.pop("time", None)
    return data

    """
    Normalize model JSON output:
    - Ensure due_date is YYYY-MM-DD
    - Ensure time is HH:MM (24h) if present
    - Ignore dates/times from 'Not available' and 'Available until" sections
    """
    data = json.loads(generated_json)

    for item in data:
        # If the assignment or quiz text contains "Not available", drop its date/time
        if ("assignment" in item or "quiz" in item) and ("not available" in item["assignment"].lower() or "available until" in item["assignment"].lower()):
            item.pop("due_date", None)
            item.pop("time", None)
            continue
        # Normalize due_date
        if "due_date" in item and item["due_date"]:
            d, t = normalize_time(item["due_date"])
            if d:
                item["due_date"] = d
            else:
                item.pop("due_date", None)
            if t and "time" not in item:
                item["time"] = t
        # Normalize time
        if "time" in item and item["time"]:
            _, t = normalize_time(item["time"])
            if t:
                item["time"] = t
            else:
                item.pop("time", None)
    return data

# -----------------------------
# FastAPI Endpoint
# -----------------------------
@app.post("/extract-assignments", response_model=Assignments)
def extract_assignments(req: ExtractRequest):
    # Step 1: mark, remove ignored lines, clean <DUE>
    print("req", req.text)
    marked = mark_tags(req.text)
    cleaned = clean_all_assignment_blocks(remove_ignore_lines(marked))
    # Step 2: create prompt
    messages = createMessages(cleaned)
    print("messages", messages)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=750,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])
    print("generated_text", generated_text)
    # Step 3: parse JSON
    assignments_list = postprocess_json(generated_text)
    return {"assignments": assignments_list}
