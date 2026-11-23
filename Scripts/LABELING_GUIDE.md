# Hand-labeling guide for gold_sample_all.csv

You are deciding, for each line in the `text` column, whether the speaker was **reading the
script** or **talking out of character**. Your labels are the "truth" the split is scored
against, so label what actually happened, not what you think the script would guess.

Before you start: do not open the `*_labeled.csv` files or look at the script's labels
while you label. Seeing them biases the result.

## What to type in `gold_label`

| Type | When |
| --- | --- |
| `in_character` (or `in`) | The words were read from the script as part of the performance. |
| `out_of_character` (or `out`) | Anything else: discussion, questions, jokes, directions, logistics. |
| `skip` | You truly cannot tell, even with the lines around it. Skipped rows are left out of the score; say how many in the paper. Try to keep this under 5%. |

Use `previous_line` and `next_line` for context. Most lines are easy; the context
columns settle the hard ones.

## Rules for the edge cases

1. **Revised lines count as in character.** The sessions read *Untitled Othello*, a
   revision. A line read from that script is `in_character` even if its wording differs from
   Shakespeare's. (This is exactly the error the evaluation should catch.)
2. **Misreads and stumbles count as in character.** "Oh, I mean, I am not what I am" while
   reading is still `in_character`.
3. **A quote inside discussion counts as out of character.** "When Iago says 'I am not what
   I am', I think he means..." is `out_of_character`, because the speaker is discussing.
4. **Mixed lines: label what the speaker was doing.** If they were reading and added a word
   ("um", "sorry"), it is `in_character`. If they were talking and quoted part of a line, it is
   `out_of_character`.
5. **Short lines: use the context.** "My lord." in the middle of a scene is `in_character`;
   "Yeah." or "Okay." between scenes is `out_of_character`.
6. **Stage directions and speaker names read aloud** ("Enter Iago", "Act two, scene one")
   are `out_of_character`, since they are not lines spoken by a character. Whatever you
   decide, apply it the same way every time and say so in the paper.
7. **Directions about the reading** ("Let's start from the top", "Who's reading Cassio?")
   are `out_of_character`.
8. **Garbled transcription:** if you can still tell it was a play line from its rhythm,
   archaic words or context, label it `in_character`; otherwise use `skip`.

## After labeling

Save as CSV (keep the same file name) and run:

```
python GoldSample.py evaluate --labeled "path/to/labeled/*_labeled.csv" --gold gold_sample_all.csv
```

For the paper, report precision, recall and F1 for the `in_character` row, plus the
number of rows labeled and skipped. Labeling 200 rows takes about 45 to 60 minutes.
