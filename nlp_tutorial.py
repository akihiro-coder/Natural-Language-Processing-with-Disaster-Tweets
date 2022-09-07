import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
# print(type(train_df))
# print(train_df)
# print('------------------------------------------------')
# print(train_df['target'])
# print('------------------------------------------------')
#print(train_df[train_df['target'] == 0])
# print('------------------------------------------------')
#print(train_df[train_df['target'] == 0]['text'])
#print(type(train_df[train_df['target'] == 0]['text']))
# print('------------------------------------------------')
#print(train_df[train_df['target'] == 0]['text'].values[1])
#
#
# print('======================================================')
#

count_vectorizer = feature_extraction.text.CountVectorizer()
# print(train_df['text'][0:5])
print(type(train_df['text']))
example_train_vectors = count_vectorizer.fit_transform(train_df['text'][0:5])
print(example_train_vectors[0].todense())


train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.fit_transform(test_df['text'])

clf = linear_model.RidgeClassifier()

#scores = model_selection.cross_val_score(clf, train_vectors, train_df['text'], cv=3, scoring='f1')
clf.fit(train_vectors, train_df['text'])

sample_submission = pd.read_csv('./data/sample_submission.csv')
sample_submission['target'] = clf.predict(test_vectors)

print(sample_submission.head())


sample_submission.to_csv('submission.csv', index=False)
