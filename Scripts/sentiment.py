import pandas as pd
import os
import nltk

from sklearn import preprocessing
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
nltk.download('opinion_lexicon')
nltk.download('punkt')

Files = folder = r"C:/Users/liamb/Source/Othello_01/Data/cleaned"
for filename in sorted(os.listdir(Files)):
    if filename.endswith("in_character.txt"):
        continue
    else:
        break

df = pd.read_csv(Files, lines = True)
df.rename(columns = {"review text": "text"}, inplace = True)

pos_score = 1
neg_score = -1
word_dict = {}

for word in opinion_lexicon.positive():
    word_dict[word] = pos_score

for word in opinion_lexicon.negative():
    word_dict[word] = neg_score

def bing_liu_score(text):
    sentiment_score = 0
    bunch_o_words = word_tokenize(text.lower())
    for word in bunch_o_words:
        if word in word_dict:
            sentiment_score += word_dict[word]
        return sentiment_score

df['text'].filna('no review', inplace = True)
df['bing_liu_score'] = df['text'].apply(bing_liu_score)

df[['overall', "text", 'bing_liu_score']].head(10)
df.groupby('overall').agg({'bing_liu_score':'mean'})