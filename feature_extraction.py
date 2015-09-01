import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

# NOTE: this is broken, will throw an error from line 7 (no data)
#       Instead, use feature_extraction_2.py
count_vect.fit(text)

print count_vect.vocabulary_.get(u'3g')
