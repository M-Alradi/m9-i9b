"""Slot extraction — fill the named slots a shape's Cypher template needs.

Each shape in `shapes.CANONICAL_CYPHER` carries `$param` placeholders.
Your `extract_slots(question, shape)` returns a dict whose keys are the
parameter names the template expects, e.g.:

  ShapeId.Q1 → {"ingredient": "ginger"}
  ShapeId.Q5 → {"cuisine": "Sichuan", "ingredient": "ginger"}
  ShapeId.Q9 → {"cuisine": "Italian"}
  ShapeId.Q10 → {"max_minutes": 30}
  ShapeId.Q14 → {"ingredient": "ginger", "exclude_ingredient": "garlic"}

See `data/eval_questions.jsonl` for the gold (question_text, shape, slots)
triples used by the autograder.
"""

import re
import spacy
from .shapes import ShapeId


nlp = spacy.load("en_core_web_sm")

# Use the actual KG vocab if available
CUISINES = {
    "italian": "Italian",
    "asian": "Asian",
    "chinese": "Chinese",
    "sichuan": "Sichuan",
}

INGREDIENTS = {
    "ginger": "ginger",
    "garlic": "garlic",
    "basil": "basil",
    "peppercorn": "peppercorn",
}

TECHNIQUES = {
    "wok": "wok",
}


def _find_cuisine(text: str) -> str | None:
    q = text.lower()
    for key, canonical in CUISINES.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            return canonical
    return None


def _find_ingredient(text: str) -> str | None:
    q = text.lower()
    for key, canonical in INGREDIENTS.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            return canonical
    return None


def _find_author(text: str) -> str | None:
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    m = re.search(r"by author\s+(.+?)(?:\s+that|\s*$)", text, re.I)
    if m:
        return m.group(1).strip()

    return None


def _find_technique(text: str) -> str | None:
    q = text.lower()

    for key, canonical in TECHNIQUES.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            return canonical

    return None

def extract_slots(question: str, shape: ShapeId) -> dict:
    """Extract slot values for the given shape from the question text.

    Suggested approach:
      - spaCy NER for PERSON entities (q2, q8 author slot).
      - A short hand-authored vocabulary list of the cuisines and
        ingredients in the recipe KG — string-match the question against
        it case-insensitively. The lists are small (16 cuisines, 40
        ingredients) so a literal-match approach is fine.
      - For q10: a regex like `under (\\d+)\\s*minutes` to pull the
        integer threshold.
      - For q14: split the question on "but not" / "without" to get the
        positive and negative ingredient slots.

    Return a dict whose keys EXACTLY match the `$param` names in
    shapes.CANONICAL_CYPHER[shape]. Returning a slot dict missing a
    required parameter will surface as a Neo4j ParameterMissing error
    at query time — that is fail-loud and desired.

    Values must be the canonical form the KG uses (e.g., 'Italian' not
    'italian'; 'ginger' not 'Ginger'). Match against the schema vocabulary
    rather than echoing the surface form of the question.
    """
    q = question.lower()

    # Q1
    if shape == ShapeId.Q1:
        return {
            "ingredient": _find_ingredient(question)
        }

    # Q2
    if shape == ShapeId.Q2:
        return {
            "author": _find_author(question)
        }

    # Q3 / Q4 / Q9
    if shape in {ShapeId.Q3, ShapeId.Q4, ShapeId.Q9}:
        return {
            "cuisine": _find_cuisine(question)
        }

    # Q5 / Q6
    if shape in {ShapeId.Q5, ShapeId.Q6}:
        return {
            "cuisine": _find_cuisine(question),
            "ingredient": _find_ingredient(question),
        }

    # Q7
    if shape == ShapeId.Q7:
        return {
            "technique": _find_technique(question)
        }

    # Q8
    if shape == ShapeId.Q8:
        return {
            "author": _find_author(question),
            "ingredient": _find_ingredient(question),
        }

    # Q10
    if shape == ShapeId.Q10:
        m = re.search(r"under\s+(\d+)\s*minutes?", q)
        return {
            "max_minutes": int(m.group(1))
        }

    # Q11
    if shape == ShapeId.Q11:
        return {
            "cuisine": _find_cuisine(question)
        }
    
    if shape == ShapeId.Q12:
        return {
            "cuisine": _find_cuisine(question)
        }

    # Q13
    if shape == ShapeId.Q13:
        return {
            "ingredient": _find_ingredient(question)
        }

    # Q14
    if shape == ShapeId.Q14:
        m = re.search(
            r"(?:use|uses|using|with)\s+(\w+).*?(?:but not|without)\s+(\w+)",
            q,
        )
        return {
            "ingredient": m.group(1),
            "exclude_ingredient": m.group(2),
        }

    # Q15
    if shape == ShapeId.Q15:
        return {
            "technique": _find_technique(question)
        }

    if shape == ShapeId.Q16:
        return {
            "cuisine": _find_cuisine(question),
            "author": _find_author(question),
            "technique": _find_technique(question),
        }

    if shape == ShapeId.Q17:
        return {
            "author": _find_author(question),
        }

    if shape == ShapeId.Q18:
        return {
            "cuisine": _find_cuisine(question),
            "technique": _find_technique(question),
        }

    if shape == ShapeId.Q19:
        return {
            "ingredient": _find_ingredient(question),
            "technique": _find_technique(question),
        }

    if shape == ShapeId.Q20:
        return {
            "cuisine": _find_cuisine(question),
            "technique": _find_technique(question),
        }

    return {}
