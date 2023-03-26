import pandas as pd
import codecs
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Read data from the HindiSentiWordnet.txt file into a Pandas dataframe
data = pd.read_csv("HindiSentiWordnet.txt", delimiter=' ')

# Define fields to be used for extracting required information from the dataframe
fields = ['POS_TAG', 'ID', 'POS', 'NEG', 'LIST_OF_WORDS']

# Create a dictionary containing words from the HindiSentiWordnet dataset and their associated sentiment information
words_dict = {}
for i in data.index:
    words = data[fields[4]][i].split(',')
    for word in words:
        words_dict[word] = (data[fields[0]][i], data[fields[2]][i], data[fields[3]][i])

# Define a function to determine the sentiment of a given text
def sentiment(text):
    words = word_tokenize(text)
    votes = []
    pos_polarity = 0
    neg_polarity = 0
    allowed_words = ['a','v','r','n']
    for word in words:
        if word in words_dict:
            pos_tag, pos, neg = words_dict[word]
            if pos_tag in allowed_words:
                if pos > neg:
                    pos_polarity += pos
                    votes.append(1)
                elif neg > pos:
                    neg_polarity += neg
                    votes.append(0)
    pos_votes = votes.count(1)
    neg_votes = votes.count(0)
    if pos_votes > neg_votes:
        return 1
    elif neg_votes > pos_votes:
        return 0
    else:
        if pos_polarity < neg_polarity:
            return 0
        else:
            return 1

# Extract sentiment for positive Hindi reviews and store the predicted and actual sentiment values
pred_y = []
actual_y = []
pos_reviews = codecs.open("pos_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
for line in pos_reviews.split('$'):
    data = line.strip('\n')
    if data:
        pred_y.append(sentiment(data))
        actual_y.append(1)

# Extract sentiment for negative Hindi reviews and store the predicted and actual sentiment values
neg_reviews = codecs.open("neg_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
for line in neg_reviews.split('$'):
    data=line.strip('\n')
    if data:
        pred_y.append(sentiment(data))
        actual_y.append(0)

# Calculate accuracy and F1-score and print the results
print(len(actual_y))
print(accuracy_score(actual_y, pred_y) * 100)
print('F-measure:  ',f1_score(actual_y,pred_y))
