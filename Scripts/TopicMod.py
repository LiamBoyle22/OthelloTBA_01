import pandas as pd
import glob
import os
import re
import pyLDAvis
import pyLDAvis.lda_model
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords = list(stopwords)

extras_stops = {
    # Shakespearean forms
    "thou","thee","thy","thine","dost","doth","art","hath","tis","twas",
    "ay","aye","nay","oh","o","ll","ve","d","s","st","wilt","shalt","ere",
    # common filler in the play
    "lord","sir","come","let","know","think","good","shall","pray","speak",
    "say","make","like","did","does","man","men","well","yes", "okay", "right", "yeah", "sure", "then","enter","exit"
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | extras_stops)

folder = r"C:/Users/liamb/Source/Othello_01/Data/cleaned"
rows = []
for path in glob.glob(os.path.join(folder, "*out_of_character.txt")):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows.append({"file": os.path.basename(path), "text": f.read()})
df = pd.DataFrame(rows)
print(len(df), "files loaded")

df["paragraphs"] = df["text"].map(lambda text: re.split(r'\.\s*\n', text))
df["number_paragraphs"] = df["paragraphs"].map(len)

paragraph_df = pd.DataFrame([{"text": paragraph, "file": file}
                             for paragraphs, file in zip(df["paragraphs"], df["file"])
                             for paragraph in paragraphs if paragraph.strip()])
print(len(paragraph_df), "paragraphs total")

def display_topics(model, features, no_top_words=5):
    for topic, words in enumerate(model.components_):
        total = words.sum()
        largest = words.argsort()[::-1]
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print(" %s (%2.2f)" % (features[largest[i]],
                                   abs(words[largest[i]] * 100.0 / total)))
#NMF
tfidf_para_vectorizer = TfidfVectorizer(stop_words = STOP_WORDS, min_df = 5, max_df = 0.7)
tfidf_para_vectors = tfidf_para_vectorizer.fit_transform(paragraph_df["text"])

nmf_para_model = NMF(n_components = 10, random_state = 42)
W_para_matrix = nmf_para_model.fit_transform(tfidf_para_vectors)
print("\nNMF topics")
display_topics(nmf_para_model, tfidf_para_vectorizer.get_feature_names_out())
#LDA
count_para_vectorizer = CountVectorizer(stop_words = STOP_WORDS, min_df = 5, max_df = 0.7)
count_para_vectors = count_para_vectorizer.fit_transform(paragraph_df["text"])

lda_para_model = LatentDirichletAllocation(n_components = 10, random_state = 42)
W_lda_para_matrix = lda_para_model.fit_transform(count_para_vectors)
print("\nLDA topics")
display_topics(lda_para_model, count_para_vectorizer.get_feature_names_out())

lda_display = pyLDAvis.lda_model.prepare(lda_para_model, count_para_vectors,
                                         count_para_vectorizer, sort_topics = False, mds="mmds")
pyLDAvis.save_html(lda_display, "lda_vis.html")

def wordcloud_topics(model, features, name, no_top_words = 40):
    for topic, words in enumerate(model.components_):
        size = {}
        largest = words.argsort()[::-1]
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color ="white", max_words = 100, width = 960, height=540)
        wc.generate_from_frequencies(size)
        plt.figure(figsize = (12, 12))
        plt.imshow(wc, interpolation ="bilinear")
        plt.axis("off")
        plt.savefig(f"{name}_topic{topic}.png")
        plt.close()

wordcloud_topics(nmf_para_model, tfidf_para_vectorizer.get_feature_names_out(), "nmf")
wordcloud_topics(lda_para_model, count_para_vectorizer.get_feature_names_out(), "lda")

topic_table = pd.DataFrame(W_para_matrix)
topic_table["file"] = paragraph_df["file"].values
by_file = topic_table.groupby("file").sum()
by_file = by_file.div(by_file.sum(axis=1), axis=0) * 100
by_file.round(1).to_csv("topics_by_file.csv")
print("\nSaved topics_by_file.csv")