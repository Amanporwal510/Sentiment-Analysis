import codecs
import random

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Hindi stopwords
stopwords = codecs.open("hindi_stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')

def get_features(type='dbn'):
    # Load positive and negative Hindi reviews
    pos_reviews = codecs.open("pos_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
    neg_reviews = codecs.open("neg_hindi.txt", "r", encoding='utf-8', errors='ignore').read()

    documents = []
    pos_count = 0
    for i in pos_reviews.split('$'):
        data = i.strip('\n')
        if data:
            documents.append(data)
            pos_count += 1

    neg_count = 0
    for i in neg_reviews.split('$'):
        data = i.strip('\n')
        if data:
            documents.append(data)
            neg_count += 1

    # Vectorize documents using TF-IDF
    vectorizer = TfidfVectorizer(lowercase=False, analyzer=word_tokenize)
    tfidf = vectorizer.fit_transform(documents)
    featureset = tfidf.toarray()

    positive_feature_set = featureset[0:pos_count]
    negative_feature_set = featureset[pos_count:]
    final_set = []

    # Create final dataset
    for i in positive_feature_set:
        if type == 'simple':
            final_set.append([i, [1, 0]])
        else:
            final_set.append([i, 1])
    for i in negative_feature_set:
        if type == 'simple':
            final_set.append([i, [0, 1]])
        else:
            final_set.append([i, 0])
    final_set = np.array(final_set)
    random.shuffle(final_set)

    # Split dataset into training and testing sets
    test_size = 0.2
    testing_size = int((1 - test_size) * len(final_set))
    x_train = list(final_set[:, 0][:testing_size])
    y_train = list(final_set[:, 1][:testing_size])
    x_test = list(final_set[:, 0][testing_size:])
    y_test = list(final_set[:, 1][testing_size:])
    return x_train, y_train, x_test, y_test

get_features()
