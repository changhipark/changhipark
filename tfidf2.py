import sys, scipy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_IN_PATH = '/Users/choeh/data_in'
TRAIN_CLEAN_DATA = 'train_clean.csv'
train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['review'])
vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True,
                             ngram_range=(1,3), max_features=5000)
X = vectorizer.fit_transform(reviews)
