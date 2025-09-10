from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
import dateparser
import json
import re

#    python -m spacy download en_core_web_trf 

##### NER ######
#nlp_token_extractorrrrrr = spacy.load("en_core_web_sm")
nlp_token_extractor = spacy.load("en_core_web_trf")

def extract_time_token_entities(text: str):
    doc = nlp_token_extractor(text)
    times_more = []
    for entity in doc.ents:
        if entity.label_ in ["TIME", "DATE"]:
            times_more.append({
                "text": entity.text,
                "label": entity.label_,
                "start": entity.start_char,
                "end": entity.end_char
            })
    return times_more

def mark_time_tokens(text: str):
    """
    Replace date/time expressions with <DATE> and <TIME> tags.
    Handles cases where SpaCy splits DATE and TIME separately.
    """
    doc = nlp_token_extractor(text)
    ents = list(doc.ents)

    merged = []
    skip_next = False
    for i, ent in enumerate(ents):
        if skip_next:
            skip_next = False
            continue

        # Merge DATE + TIME spans like "Sep 8 at 11:59pm"
        if ent.label_ == "DATE" and i + 1 < len(ents) and ents[i + 1].label_ == "TIME":
            merged_text = text[ent.start_char:ents[i + 1].end_char]
            merged.append({
                "text": merged_text,
                "start": ent.start_char,
                "end": ents[i + 1].end_char
            })
            skip_next = True
        else:
            merged.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})

    # Replace in text
    marked_text = text
    offset = 0
    for m in merged:
        date_str, time_str = normalize_time(m["text"])
        if date_str or time_str:
            start = m["start"] + offset
            end = m["end"] + offset

            if date_str and time_str:
                replacement = f"<DATE>{date_str}</DATE> <TIME>{time_str}</TIME>"
            elif date_str:
                replacement = f"<DATE>{date_str}</DATE>"
            elif time_str:
                replacement = f"<TIME>{time_str}</TIME>"
            else:
                continue

            marked_text = marked_text[:start] + replacement + marked_text[end:]
            offset += len(replacement) - (end - start)

    return marked_text
    """
    Mark assignments text with NER tags:
    - <ASSIGNMENT>...</ASSIGNMENT> around each assignment block
    - <IGNORE>...</IGNORE> for 'Not available' lines
    - <DATE>YYYY-MM-DD</DATE> for dates
    - <TIME>HH:MM</TIME> for times
    - Handles cases where SpaCy splits DATE and TIME separately.
    """
    # split by assignment blocks (assuming each starts with '<ASSIGNMENT>' or similar)
    blocks = re.split(r'(?:Assignment|Quiz)\b', text)

    marked_blocks = []
    for block in blocks:
        if not block.strip():
            continue
        # wrap 'Not available' lines with <IGNORE>
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if "not available" in line.lower():
                lines[i] = f"<IGNORE>{line}</IGNORE>"
        block_text = "\n".join(lines)
        # extract nentitiesr with SpaCy
        doc = nlp_token_extractor(block_text)
        ents = list(doc.ents)
        # merge DATE + TIME spans
        merged = []
        skip_next = False
        for i, ent in enumerate(ents):
            if skip_next:
                skip_next = False
                continue
            # skip entities inside <IGNORE>
            if re.search(r"<IGNORE>.*?</IGNORE>", block_text[ent.start_char:ent.end_char]):
                continue

            if ent.label_ == "DATE" and i + 1 < len(ents) and ents[i + 1].label_ == "TIME":
                merged_text = block_text[ent.start_char:ents[i + 1].end_char]
                merged.append({
                    "text": merged_text,
                    "start": ent.start_char,
                    "end": ents[i + 1].end_char
                })
                skip_next = True
            else:
                merged.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})
        # replace in block with <DATE> / <TIME> tags
        marked_block = block_text
        offset = 0
        for m in merged:
            date_str, time_str = normalize_time(m["text"])
            if not date_str and not time_str:
                continue
            start = m["start"] + offset
            end = m["end"] + offset
            if date_str and time_str:
                replacement = f"<DATE>{date_str}</DATE> <TIME>{time_str}</TIME>"
            elif date_str:
                replacement = f"<DATE>{date_str}</DATE>"
            elif time_str:
                replacement = f"<TIME>{time_str}</TIME>"
            marked_block = marked_block[:start] + replacement + marked_block[end:]
            offset += len(replacement) - (end - start)
        #  wrap block with <ASSIGNMENT>
        marked_blocks.append(f"<ASSIGNMENT>{marked_block}</ASSIGNMENT>")
    return "\n".join(marked_blocks)

def normalize_time(text: str):
    """Normalize raw time expressions into (date, time)."""
    dt = dateparser.parse(text)
    if not dt:
        return None, None
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M") if (dt.hour or dt.minute) else None
    return date_str, time_str

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
            print("Checking entity:", ent.text, ent.label_, "in text:", block_text[ent.start_char:ent.end_char])
            print("block text:", block_text)
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

import re

def remove_ignore_lines(text: str) -> str:
    """
    Removes all lines enclosed in <IGNORE>...</IGNORE> tags,
    including the tags themselves.
    """
    # remove everything inside <IGNORE>...</IGNORE>
    cleaned_text = re.sub(r"<IGNORE>.*?</IGNORE>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = "\n".join([line for line in cleaned_text.split("\n") if line.strip()])
    return cleaned_text

def clean_due_tags(block_text: str):
    """
    Keep only the first <DUE> tag that comes after the keyword 'Due'.
    Remove all other <DUE> tags in the block.
    """
    # Search for the 'Due' keyword
    match = re.search(r'\bDue\b', block_text, flags=re.IGNORECASE)
    if not match:
        return block_text
    before_due = block_text[:match.end()]
    after_due = block_text[match.end():]
    # Find all <DUE> tags in the text after 'Due'
    due_tags = re.findall(r'<DUE>.*?</DUE>', after_due)
    if not due_tags:
        return block_text
    # Keep only the first <DUE>
    first_due = due_tags[0]
    # Remove all <DUE> tags in the remainder
    after_due_cleaned = re.sub(r'<DUE>.*?</DUE>', '', after_due)
    # Reinsert only the first <DUE>
    after_due_cleaned = first_due + after_due_cleaned
    return before_due + after_due_cleaned

def clean_all_assignment_blocks(marked_text: str):
    """
    Apply clean_due_tags to each <ASSIGNMENT> block in the marked text.
    """
    def clean_block(match):
        block = match.group(0)
        return clean_due_tags(block)
    cleaned_text = re.sub(r'<ASSIGNMENT>.*?</ASSIGNMENT>', clean_block, marked_text, flags=re.DOTALL)
    return cleaned_text

def postprocess_json(generated_json: str, raw_text: str = ""):
    """
    Normalize model JSON output:
    - Ensure due_date is YYYY-MM-DD
    - Ensure time is HH:MM (24h) if present
    - Ignore dates/times from 'Not available' sections
    """
    if not generated_json.strip():
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
    - Ignore dates/times from 'Not available' sections
    """
    data = json.loads(generated_json)
    for item in data:
        # If the assignment text contains "Not available", drop its date/time
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



# Initialize model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

assignments = """
Assignment
Professional emails 
Due Sep 8 at 11:59pm Sep 8 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
AI and education 
Not available until Sep 3 at 12pm Sep 3 at 12pm
Due Sep 9 at 11:59pm Sep 9 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
AI and relationships 
Not available until Sep 3 at 12pm Sep 3 at 12pm
Due Sep 9 at 11:59pm Sep 9 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
AI and employment 
Not available until Sep 3 at 12pm Sep 3 at 12pm
Due Sep 9 at 11:59pm Sep 9 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
Architecture diagram (prep) 
Due Sep 16 at 11:59pm Sep 16 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
Readme 
Due Sep 16 at 11:59pm Sep 16 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
Requirements (prep) 
Not available until Sep 15 at 12am Sep 15 at 12am
Due Sep 23 at 11:59pm Sep 23 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
Reddit comment 
Not available until Sep 22 at 12am Sep 22 at 12am
Due Sep 30 at 11:59pm Sep 30 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
Medium post (prep) 
Not available until Sep 15 at 12am Sep 15 at 12am
Due Sep 30 at 11:59pm Sep 30 at 11:59pm
-/4 ptsNo submission for this assignment. 4 points possible.
Assignment
Usability & accessibility (prep) 
Not available until Oct 6 at 12am Oct 6 at 12am
Due Oct 14 at 11:59pm Oct 14 at 11:59pm
-/0 ptsNo submission for this assignment. 0 points possible.
Assignment
Lightning Presentation 
Not available until Oct 13 at 12am Oct 13 at 12am
Due Nov 4 at 11:59pm Nov 4 at 11:59pm
-/16 ptsNo submission for this assignment. 16 points possible.
Assignment
Final Portfolio 
Due Dec 8 at 11:59pm Dec 8 at 11:59pm
-/32 pts
"""


# assignments = """
# Quiz
# Quiz #1: Syllabus and Basics 
# Not available until Sep 4 at 5pm Sep 4 at 5pm
# Due Sep 8 at 11:59pm Sep 8 at 11:59pm
# -/20 ptsNo submission for this assignment. 20 points possible.
# Assignment
# HW1: Encode and decode messages 
# Available until Sep 15 at 11:59pm Sep 15 at 11:59pm
# Due Sep 11 at 11:59pm Sep 11 at 11:59pm
# -/50 ptsNo submission for this assignment. 50 points possible.
# Assignment
# HW2: Invent two new image filters and build a collage 
# Not available until Sep 9 at 5pm Sep 9 at 5pm
# Due Sep 18 at 11:59pm Sep 18 at 11:59pm
# -/60 ptsNo submission for this assignment. 60 points possible.
# Quiz
# Quiz #2: Images 
# Not available until Sep 16 at 5pm Sep 16 at 5pm
# Due Sep 20 at 11:59pm Sep 20 at 11:59pm
# -/20 ptsNo submission for this assignment. 20 points possible.
# Quiz
# Quiz #3: Copying and Transforming Pictures 
# Not available until Sep 25 at 5pm Sep 25 at 5pm
# Due Sep 28 at 11:59pm Sep 28 at 11:59pm
# -/20 ptsNo submission for this assignment. 20 points possible.
# Assignment
# Project 1: Build an image collage 
# Not available until Sep 16 at 10am Sep 16 at 10am
# Due Oct 7 at 11:59pm Oct 7 at 11:59pm
# -/55 ptsNo submission for this assignment. 55 points possible.
# Quiz
# Quiz #4: Sound Basics 
# Not available until Oct 9 at 5pm Oct 9 at 5pm
# Due Oct 12 at 11:59pm 
# """

assignments2 = """
Quiz #1: Syllabus and Basics 
Not available until Sep 4 at 5pm Sep 4 at 5pm
Due Sep 8 at 11:59pm Sep 8 at 11:59pm
-/20 ptsNo submission for this assignment. 20 points possible.
Assignment
HW1: Encode and decode messages 
Available until Sep 15 at 11:59pm Sep 15 at 11:59pm
Due Sep 11 at 11:59pm Sep 11 at 11:59pm
-/50 ptsNo submission for this assignment. 50 points possible.
Assignment
HW2: Invent two new image filters and build a collage 
Not available until Sep 9 at 5pm Sep 9 at 5pm
Due Sep 18 at 11:59pm Sep 18 at 11:59pm
-/60 ptsNo submission for this assignment. 60 points possible.
Quiz
Quiz #2: Images 
Not available until Sep 16 at 5pm Sep 16 at 5pm
Due Sep 20 at 11:59pm Sep 20 at 11:59pm
-/20 ptsNo submission for this assignment. 20 points possible.
Quiz
Quiz #3: Copying and Transforming Pictures 
Not available until Sep 25 at 5pm Sep 25 at 5pm
Due Sep 28 at 11:59pm Sep 28 at 11:59pm
-/20 ptsNo submission for this assignment. 20 points possible.
Assignment
Project 1: Build an image collage 
Not available until Sep 16 at 10am Sep 16 at 10am
Due Oct 7 at 11:59pm Oct 7 at 11:59pm
-/55 ptsNo submission for this assignment. 55 points possible.
Quiz
Quiz #4: Sound Basics 
Not available until Oct 9 at 5pm Oct 9 at 5pm
Due Oct 12 at 11:59pm Oct 12 at 11:59pm
-/20 ptsNo submission for this assignment. 20 points possible.
Assignment
HW3: Create two sound filters 
Not available until Oct 7 at 12am Oct 7 at 12am
Due Oct 21 at 11:59pm Oct 21 at 11:59pm
-/50 ptsNo submission for this assignment. 50 points possible.
Quiz
Quiz #5: Advanced Sound 
Not available until Oct 30 at 5pm Oct 30 at 5pm
Due Nov 2 at 11:59pm Nov 2 at 11:59pm
-/20 pts
"""

# clean input 
print("Extracted time tokens:", extract_time_token_entities(assignments))
marked_assignments = mark_tags(assignments2)
print("Marked text:", marked_assignments)
cleaned_assignments = clean_all_assignment_blocks(remove_ignore_lines(marked_assignments))
print("Cleaned text:", cleaned_assignments)

# prompt
messages = createMessages(cleaned_assignments)
print("Messages:", messages)
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1250,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])
print("Raw model output:", generated_text)

# clean json
def postprocessss_json(generated_json: str):
    data = json.loads(generated_json)
    for item in data:
        if "due_date" in item:
            date_str, time_str = normalize_time(item["due_date"])
            if date_str:
                item["due_date"] = date_str
            if time_str:
                item["time"] = time_str
            elif "time" in item:
                item.pop("time") 
    return data

cleaned = postprocess_json(generated_text)
print("Final output:", cleaned)