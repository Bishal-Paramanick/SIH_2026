"""
Entity extraction from FIR text files
- Loads a FIR text file
- Runs spaCy NER for PERSON, GPE (locations), ORG
- Runs custom regex for phone numbers and vehicle plates
- Prints results
"""

import re
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_trf")

# ---- Custom regex patterns ----
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")                       # Indian mobile numbers
VEHICLE_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b")   # e.g. WB01AB1234
ACCOUNT_PATTERN = re.compile(r"\b\d{11,16}\b")                      # bank account numbers
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")     # dd/mm/yyyy style

# Generic fallback for organization names spaCy's NER sometimes misses.
# Matches capitalized phrases ending in a common business-type word —
# this is a STRUCTURAL pattern (works for any org name), not a memorized
# list of specific companies.
ORG_SUFFIXES = (
    r"Supermarket|Traders|Electronics|Enterprises|Textiles|Stores|Finance|"
    r"Logistics|Bank|Industries|Motors|Pharma|Foods|Exports|Imports|"
    r"Corporation|Company|Associates|Solutions|Services"
)
ORG_FALLBACK_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z&]+(?:\s[A-Z][a-zA-Z&.]+){0,3}\s(?:" + ORG_SUFFIXES + r")\b"
    r"(?:\s(?:Pvt\.?\s?Ltd\.?|Ltd\.?|Limited))?)"
)

# Vehicle plate pattern (kept as VEHICLE type via regex, not brand names)


def clean_text(text: str) -> str:
    """Collapse newlines/extra whitespace so entities don't leak line breaks."""
    return re.sub(r"\s+", " ", text)


def clean_entity_name(name: str) -> str:
    """Strip possessives, relationship prefixes, and surrounding whitespace."""
    name = name.strip()
    name = re.sub(r"['’]s\b", "", name)  # strip possessive 's

    # Strip Indian FIR relationship prefixes: D/o, S/o, W/o, C/o, R/o
    # (Daughter of, Son of, Wife of, Care of, Resident of)
    name = re.sub(r"^(D/o|S/o|W/o|C/o|R/o)\s+", "", name, flags=re.IGNORECASE)

    return name.strip()


def extract_entities_from_text(text: str, doc_id: str, debug_labels: bool = False):
    """
    Runs spaCy NER + regex on a block of text.
    Returns a list of entity dicts (before deduplication/ID assignment).

    debug_labels: if True, prints spaCy's raw label for every entity
    (useful for checking what label vehicle brand names get tagged as,
    instead of guessing with a hardcoded brand list).
    """
    text = clean_text(text)
    entities = []

    # Find date spans so we can skip anything spaCy tags as GPE/LOC that
    # actually overlaps a date (dates were getting misclassified as LOCATION)
    date_spans = [m.span() for m in DATE_PATTERN.finditer(text)]

    def overlaps_date(start, end):
        return any(start < d_end and end > d_start for d_start, d_end in date_spans)

    # --- spaCy NER pass ---
    doc = nlp(text)
    for ent in doc.ents:
        if debug_labels:
            print(f"    (raw label: {ent.label_:10} text: {ent.text!r})")

        name = clean_entity_name(ent.text)
        if not name or overlaps_date(ent.start_char, ent.end_char):
            continue

        if ent.label_ == "PERSON":
            entities.append({"name": name, "type": "PERSON", "doc_id": doc_id})
        elif ent.label_ in ("GPE", "LOC"):
            entities.append({"name": name, "type": "LOCATION", "doc_id": doc_id})
        elif ent.label_ == "PRODUCT":
            continue  # generic filter: vehicle brands/models are usually tagged PRODUCT, not ORG
        elif ent.label_ == "ORG":
            # Skip anything that's actually a vehicle plate mis-tagged as ORG
            if VEHICLE_PATTERN.fullmatch(name):
                continue
            entities.append({"name": name, "type": "ORG", "doc_id": doc_id})

    # --- Regex passes (spaCy won't reliably catch these) ---
    for match in PHONE_PATTERN.findall(text):
        entities.append({"name": match, "type": "PHONE", "doc_id": doc_id})

    for match in VEHICLE_PATTERN.findall(text):
        entities.append({"name": match, "type": "VEHICLE", "doc_id": doc_id})

    for match in ACCOUNT_PATTERN.findall(text):
        entities.append({"name": match, "type": "ACCOUNT", "doc_id": doc_id})

    # Fallback: catch org names spaCy's NER missed, using structural suffix match
    already_found = {e["name"] for e in entities if e["type"] == "ORG"}
    for match in ORG_FALLBACK_PATTERN.findall(text):
        if match not in already_found:
            entities.append({"name": match, "type": "ORG", "doc_id": doc_id})
            already_found.add(match)

    return entities


def load_and_extract(filepath: str, doc_id: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_entities_from_text(text, doc_id)


if __name__ == "__main__":
    # Update these paths/doc_ids to match your actual files
    files = [
        ("data/raw/fir_101.txt", "FIR_101"),
        ("data/raw/fir_102.txt", "FIR_102"),
        ("data/raw/fir_103.txt", "FIR_103"),
        ("data/raw/fir_104.txt", "FIR_104"),
    ]

    all_entities = []
    for filepath, doc_id in files:
        print(f"\n--- Extracting from {doc_id} ({filepath}) ---")
        try:
            entities = load_and_extract(filepath, doc_id)
        except FileNotFoundError:
            print(f"  File not found, skipping: {filepath}")
            continue

        for e in entities:
            print(f"  [{e['type']:8}] {e['name']}")

        all_entities.extend(entities)

    print(f"\nTotal entities extracted (before dedup): {len(all_entities)}")

    # Quick dedup preview (name+type), just to see unique entity count
    unique = {(e["name"], e["type"]) for e in all_entities}
    print(f"Unique (name, type) pairs: {len(unique)}")