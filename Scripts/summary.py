import networkx as nx
import numpy as np
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import re

with open(r"C:/Users/liamb/Source/Othello_01/Data/cleaned/video_split2_in_character.txt","r", encoding="utf-8") as f:
    lines = f.readlines()

sentences = []
for line in lines:
    words = re.sub(r"[^a-zA-Z]", " ", line).split()
    if words:
        sentences.append(words)

print(len(sentences))
for s in sentences[:20]:
    print(s)

def sentence_similarity(sent1, sent2):
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)
    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1,vector2)

similarity_matrix = np.zeros((len(sentences), len(sentences)))

for idx1 in range (len(sentences)):
    for idx2 in range (len(sentences)):
        if idx1 == idx2: 
            continue
        similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])

print("similarity natrix\n", similarity_matrix)

sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(sentence_similarity_graph)
print("scores", scores)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse= True)
print("indexes of top ranked_sentences order are \n\n", ranked_sentences)

n = 10
summarize_text = []
for i in range(n):
    summarize_text.append(" ".join(ranked_sentences[i][1]))

print("Summarize Text: \n", ".".join(summarize_text))