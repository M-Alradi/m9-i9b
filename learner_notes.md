# Integration 9B — Learner Notes

## 1. Intents you handled and how you classified them

I implemented `detect_shape` using simple keyword and pattern matching rules. The classifier first converts the question to lowercase and then checks for specific phrases in a priority order. More specific question types are checked before more general ones to avoid incorrect matches.

Some shapes were easy to identify. For example:

- "Find recipes that use ginger" → Q1
- "Find recipes by author Maria Rossi" → Q2
- "Find recipes with prep time under 30 minutes" → Q10

The more difficult cases were questions that could match multiple patterns. For example:

"Find Chinese recipes that use ginger"

This could potentially match both Q5 (cuisine + ingredient) and Q6 (cuisine hierarchy + ingredient). I handled this by treating cuisines such as Chinese and Asian as hierarchical cuisines and checking for those cases before the direct cuisine match.

Another ambiguous example was:

"Find recipes that use ginger but not garlic"

This contains the same ingredient cue as Q1, but it should be classified as Q14 because of the "but not" condition. To handle this correctly, I placed the Q14 rule before Q1 in the priority order.

---

## 2. A question that worked end-to-end

Question:

"Find recipes that use ginger"

Pipeline execution:

### detect_shape

Returned:

```python
ShapeId.Q1