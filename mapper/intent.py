"""Intent classifier — map NL question to a canonical ShapeId.

This file is your responsibility. Read the 15 supported shapes in
`shapes.ShapeId` and `shapes.CANONICAL_CYPHER`, then implement
`detect_shape` so that each of the 15 canonical eval questions in
`data/eval_questions.jsonl` is classified to the gold shape, and
adversarial / off-template questions return None.

The deterministic mapper is the production-discipline arm of M9B; a
classifier that returns the wrong shape on a supported question is a
real bug, and a classifier that returns a confident answer on an
off-template question is the silent-failure mode the Reading warns
against. Prefer None over a false positive.
"""

import re

from .shapes import ShapeId


def detect_shape(question: str) -> ShapeId | None:
    q = question.lower().strip()

    # --------------------------------------------------
    # Helper signals
    # --------------------------------------------------

    has_author = (
    "by author" in q
    or re.search(r"\brecipes\s+by\s+[A-Z]", question) is not None
    )

    has_ingredient_cue = (
        re.search(r"\b(use|uses|using|with)\b", q) is not None
    )

    has_require = (
        re.search(r"\brequire[s]?\b", q) is not None
    )

    has_cuisine = any(
        cuisine in q
        for cuisine in (
            "italian",
            "asian",
            "chinese",
            "sichuan",
        )
    )

    has_hierarchical_cuisine = any(
        cuisine in q
        for cuisine in (
            "asian",
            "chinese",
        )
    )

    has_popularity = (
        "ranked by popularity" in q
        or "most popular" in q
    )

    # --------------------------------------------------
    # New shapes (Q16-Q20)
    # --------------------------------------------------

    # Q16: cuisine + author + technique
    if has_author and has_cuisine and has_require:
        return ShapeId.Q16

    # Q17: author + popularity
    if has_author and has_popularity:
        return ShapeId.Q17

    # Q20: cuisine subtree + technique
    if has_hierarchical_cuisine and has_require:
        return ShapeId.Q20

    # Q18: cuisine + technique
    if has_cuisine and has_require:
        return ShapeId.Q18

    # Q19: ingredient + technique
    if has_ingredient_cue and has_require:
        return ShapeId.Q19

    # --------------------------------------------------
    # Original shapes (Q1-Q15)
    # --------------------------------------------------

    # Q14
    if "but not" in q or "without" in q:
        if has_ingredient_cue:
            return ShapeId.Q14

    # Q15
    if "optionally tagged" in q:
        return ShapeId.Q15

    # Q13
    if "or any subtype" in q or "or any kind" in q:
        return ShapeId.Q13

    # Q11
    if (
        q.startswith("find ingredients")
        or "ingredients used in" in q
    ):
        return ShapeId.Q11

    # Q12
    if (
        q.startswith("find authors of")
        or "authors of" in q
    ):
        return ShapeId.Q12

    # Q10
    if re.search(r"under\s+\d+\s+minutes?", q):
        return ShapeId.Q10

    # Q9
    if has_popularity:
        return ShapeId.Q9

    # Q8
    if has_author and has_ingredient_cue:
        return ShapeId.Q8

    # Q7
    if has_require:
        return ShapeId.Q7

    # Q6
    if has_hierarchical_cuisine and has_ingredient_cue:
        return ShapeId.Q6

    # Q5
    if has_cuisine and has_ingredient_cue:
        return ShapeId.Q5

    # Q2
    if has_author:
        return ShapeId.Q2

    # Q4
    if has_hierarchical_cuisine:
        return ShapeId.Q4

    # Q3
    if (
        "italian" in q
        or "sichuan" in q
    ):
        return ShapeId.Q3

    # Q1
    if has_ingredient_cue:
        return ShapeId.Q1

    # Off-template
    return None