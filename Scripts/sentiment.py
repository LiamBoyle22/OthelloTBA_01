import pandas as pd
import os
import nltk

from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
nltk.download('opinion_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')

folder = r"C:/Users/liamb/Source/Othello_01/Data/cleaned"
results_dir = r"C:/Users/liamb/Source/Othello_01/Data/results"
per_file_dir = os.path.join(results_dir, "per_file")

rows = []
for filename in sorted(os.listdir(folder)):
    if not filename.endswith("in_character.txt"):
        continue
    with open(os.path.join(folder, filename), encoding = "utf-8") as f:
        rows.append({
            "file": filename,
            "in_character": filename.endswith("in_character.txt"),
            "text": f.read()
        })

df = pd.DataFrame(rows)

word_dict = {w: 1 for w in opinion_lexicon.positive()}
word_dict.update({w: -1 for w in opinion_lexicon.negative()})

def bing_liu_score(text):
    tokens = word_tokenize(text.lower())
    return sum(word_dict.get(t, 0) for t in tokens)

df = df[df["text"].str.strip() != ""].copy()
df['bing_liu_score'] = df['text'].apply(bing_liu_score)

df["n_tokens"] = df["text"].apply(lambda t: len(word_tokenize(t)))
df["score_per_1k"] = df["bing_liu_score"] / df["n_tokens"] * 1000

os.makedirs(per_file_dir, exist_ok=True)

for _, row in df.iterrows():
    lines = [l.strip() for l in row["text"].splitlines() if l.strip()]
    line_df = pd.DataFrame({"line_num": range(1, len(lines) + 1), "line": lines})
    line_df["bing_liu_score"] = line_df["line"].apply(bing_liu_score)
    line_df["n_tokens"] = line_df["line"].apply(lambda t: len(word_tokenize(t)))

    out_name = os.path.splitext(row["file"])[0] + "_sentiment.csv"
    line_df.to_csv(os.path.join(per_file_dir, out_name), index=False)

df.drop(columns="text").to_csv(os.path.join(results_dir, "bing_liu_scores.csv"), index=False)