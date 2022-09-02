import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing



train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
print(train_df)

print(train_df['target'])
print(train_df[train_df['target'] == 0])
print(train_df[train_df['target'] == 0]['text'].values[1])
print(train_df[train_df['target'] == 1]['text'].values[1])


print('------------------------------------------------')

count_vectorizer =feature_extraction.text.CountVectorizer()
print(train_df['text'][0:5])
example_train_vectors = count_vectorizer.fit_transform(train_df['text'][0:5])


print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())
