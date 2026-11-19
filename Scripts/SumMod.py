import re
import os
import csv
import nltk
import numpy as np
import networkx as nx
from nltk.cluster import cosine_distance
from nltk.corpus import stopwords

nltk.download("stopwords", quiet = True)
stops = set(stopwords.words("english"))

folder = r"C:/Users/liamb/Source/Othello_01/Data/cleaned"
num_sent = 10
results = []

def sent_sim(sent1, sent2):
    sent1 = [w.lower() for w in sent1 if w.lower() not in stops]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stops]
    AllWords = list(set(sent1 + sent2))
    vec1 = [0] * len(AllWords)
    vec2 = [0] * len(AllWords)
    for w in sent1:
        vec1[AllWords.index(w)] += 1
    for w in sent2:
        vec2[AllWords.index(w)] += 1
    if not any(vec1) or not any(vec2):
        return 0
    return 1- cosine_distance(vec1,vec2)

for filename in sorted(os.listdir(folder)):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(folder, filename), "r", encoding = "utf-8") as f:
        lines = f.readlines()

    sentences = []
    original = []
    for line in lines:
        words = re.sub(r"[^a-zA-Z]", " ", line).split()
        if words:
            sentences.append(words)
            original.append(line.strip())

    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_matrix[i][j] = sent_sim(sentences[i], sentences[j])

    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, max_iter=500)
    top = sorted(scores, key = scores.get, reverse = True)[:num_sent]
    summary = " ".join(original[i] for i in sorted(top))
    results.append([filename, len(sentences), summary])

with open("summaries.csv", "w", newline="", encoding = "utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "total_sentences", "summary"])
    writer.writerows(results)

print ("done")