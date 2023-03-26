import numpy as np
import random
from collections import Counter
from googletrans import Translator
from nltk.tokenize import word_tokenize
import codecs
from dbn_outside.dbn.tensorflow import SupervisedDBNClassification

hmLines = 5000000
translator = Translator()
stopwords = codecs.open("hindi_stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')

def implementing_feature_set_and_labels(plus, minus, ts=0.2):
    lexicon = implement_lexicon(plus, minus)
    attributes = []
    attributes += handle_samples(plus, lexicon, 1)
    attributes += handle_samples(minus, lexicon, 0)
    random.shuffle(attributes)
    attributes = np.array(attributes)
    testing_size = int((1 - ts) * len(attributes))

    training_x = list(attributes[:, 0][:testing_size])  
    training_y = list(attributes[:, 1][:testing_size]) 

    test_x = list(attributes[:, 0][testing_size:])
    test_y = list(attributes[:, 1][testing_size:])
    return training_x, training_y, test_x, test_y

def handle_samples(sample, lexicon, classification):
    featureset = []
    with codecs.open(sample, 'r', encoding="utf8",errors='ignore') as f:
        contents = f.read()
        for line in contents.split('$'):
            data = line.strip('\n')
            if data:
                set_of_words = word_tokenize(data)
                set_of_words_new = []
                for w in set_of_words:
                    if not w in stopwords:
                        set_of_words_new.append(w)
                attributes = np.zeros(len(lexicon))
                for word in set_of_words_new:
                    if word in lexicon:
                        idx = lexicon.index(w)
                        attributes[idx] = 1
                attributes = list(attributes)
                featureset.append([attributes, classification])
    return featureset

def implement_lexicon(plus, minus):
    lexicon = []
    for file_name in [plus, minus]:
        with codecs.open(file_name, 'r',encoding='utf-8',errors='ignore') as f:
            contents = f.read()
            for content in contents.split('$'):
                data = content.strip('\n')
                if data:
                    set_of_words = word_tokenize(data)
                    lexicon += list(set_of_words)
    lexicons = []
    for w in lexicon:
        if not w in stopwords:
            lexicons.append(w)
    frequency_word = Counter(lexicons)  # it will return kind of dictionary
    l2 = []
    for word in frequency_word:
        if 60 > frequency_word[word]:
            l2.append(word)
    return l2




def checking_the_class(text, lexicon):
    line = translator.translate(text, dest='hi').text
    classifier = SupervisedDBNClassification.load('dbn.pkl')
    predictor = []
    set_of_words = word_tokenize(line)
    attributes = np.zeros(len(lexicon))
    for word in set_of_words:
        if word in lexicon:
            idx = lexicon.index(word)
            attributes[idx] += 1
    attributes = list(attributes)
    predictor.append(attributes)
    predictor = np.array(predictor, dtype=np.float32)
    predictor = classifier.predict(predictor)


def create_feature_set_and_labels_simple(plus, minus, ts=0.2):
    lexicon = implement_lexicon(plus, minus)
    attributes = []
    attributes += handle_samples(plus, lexicon, [1, 0])
    attributes += handle_samples(minus, lexicon, [0, 1])
    random.shuffle(attributes)
    attributes = np.array(attributes)
    testing_size = int((1 - ts) * len(attributes))

    training_x = list(attributes[:, 0][:testing_size])
    training_y = list(attributes[:, 1][:testing_size])

    test_x = list(attributes[:, 0][testing_size:])
    test_y = list(attributes[:, 1][testing_size:])
    return training_x, training_y, test_x, test_y


if _name_ == '_main_':
    implement_lexicon('pos_hindi.txt', 'neg_hindi.txt')