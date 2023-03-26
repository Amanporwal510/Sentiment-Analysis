from UTFG import create_feature_set_and_labels as create_tf_feature_set
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


from UTIFG import get_features as create_tfidf_feature_set




def begin_test(train_data, test_data, train_labels, test_labels):
    data = train_data + test_data
    labels = train_labels + test_labels

    lin_reg_clf = LinearRegression()
    log_reg_clf = LogisticRegression()
    sgd_clf = SGDClassifier()
    svc_clf = SVC()
    knn_clf = KNeighborsClassifier()
    mlp_clf = MLPClassifier()
    dt_clf = DecisionTreeClassifier()
    nb_clf = MultinomialNB()

    ensemble_clf = VotingClassifier(
        estimators=[('logr', log_reg_clf), ('sgd', sgd_clf), ('svm', svc_clf), ('kn', knn_clf), ('nn', mlp_clf), ('dt', dt_clf)],
        voting='hard')

    label_clf_pairs = [    ('LogisticRegression', log_reg_clf),    ('SGDClassifier', sgd_clf),    ('SVC', svc_clf),    ('KNeighborsClassifier', knn_clf),    ('NeuralNetwork', mlp_clf),    ('DecisionTree', dt_clf),    ('MultinomialNB', nb_clf),    ('Ensemble', ensemble_clf)]

i = 0
while i < len(label_clf_pairs):
    label, clf = label_clf_pairs[i]
    accuracy_scores = cross_val_score(clf, data, labels, cv=10, scoring='accuracy')
    f_measure_scores = cross_val_score(clf, data, labels, cv=10, scoring='f1')
    print(label, "Accuracy:  ", accuracy_scores.mean(), "+/- ", accuracy_scores.std())
    print(label, "F-measure:  ", f_measure_scores.mean())
    i += 1


def test_by_tfidf():
    train_data, train_labels, test_data, test_labels = create_tfidf_feature_set()
    begin_test(train_data, test_data, train_labels, test_labels)


def test_by_tf():
    train_data, train_labels, test_data, test_labels = create_tf_feature_set('pos_hindi.txt', 'neg_hindi.txt')
    begin_test(train_data, test_data, train_labels, test_labels)




if _name_ == '_main_':
    print("="*10)
    print(“Term Frequency(TF) Accuracies along with Unigram”)
    test_by_tf()
    print("=" * 10)
    print("Term Frequency - Inverse Document Frequencies(TFIDF) Accuracies along with Unigram")
    test_by_tfidf()