#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:27:02 2019

@author: HP
"""
# importing libraries
import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from urllib import request
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import cross_val_score


# Data preparation and preprocess
def custom_preprocessor(text):
    print("hiiii")
    print('inside first if')
    text = re.sub(r'\W+|\d+|_', ' ', text)  # removing numbers and punctuations
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces into a single space
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove a single character
    text = text.lower()
    text = nltk.word_tokenize(text)  # tokenizing
    text = [word for word in text if not word in stop_words]  # English Stopwords
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatising
    return text


filepath_dict = {'Book1': 'https://www.gutenberg.org/files/58764/58764-0.txt',
                 'Book2': 'https://www.gutenberg.org/files/58751/58751-0.txt',
                 'Book3': 'http://www.gutenberg.org/cache/epub/345/pg345.txt'}

for key, value in filepath_dict.items():
    if (key == "Book1"):
        bookLoc = filepath_dict[key]
        response = request.urlopen(bookLoc)
        raw = response.read().decode('utf-8')
        len(raw)
        first_book = custom_preprocessor(raw)
        len(first_book)
    elif (key == "Book2"):
        bookLoc = filepath_dict[key]
        response = request.urlopen(bookLoc)
        raw = response.read().decode('utf-8')
        len(raw)
        second_book = custom_preprocessor(raw)
    elif (key == "Book3"):
        bookLoc = filepath_dict[key]
        response = request.urlopen(bookLoc)
        raw = response.read().decode('utf-8')
        len(raw)
        third_book = custom_preprocessor(raw)
    else:
        pass

# Building First Book
first_book_text = ' '.join(first_book)
fileLoc = 'C:\\Users\\Hp\\EBC7100\\FirstBook\\a.txt'
with open(fileLoc, 'a', encoding="utf-8") as fout:
    fout.write(first_book_text)

# Building Second Book
second_book_text = ' '.join(second_book)
fileLoc = 'C:\\Users\\Hp\\EBC7100\\SecondBook\\b.txt'
with open(fileLoc, 'a', encoding="utf-8") as fout:
    fout.write(second_book_text)

# Building Third Book
third_book_text = ' '.join(third_book)
fileLoc = 'C:\\Users\\Hp\\EBC7100\\ThirdBook\\c.txt'
with open(fileLoc, 'a', encoding="utf-8") as fout:
    fout.write(third_book_text)


# labeling
# Cretaing tuple

# aBooklist = []

def readAtxtfile(bookText, docs, labels):
    x = 0
    i = 0
    n = 150
    while x < 200:
        temp = ""
        words = bookText.split(" ")[i:n]
        # print ("----->",words)
        for word in words:
            temp = word + " " + temp
        # temp = temp + ',a'
        # global aBooklist
        docs.append(temp)
        labels.append(0)
        i += 150
        n += 150
        x += 1

    return docs, labels


# Cretaing tuple
# bBooklist = []


def readBtxtfile(bookText, docs, labels):
    x = 0
    i = 0
    n = 150
    while x < 184:
        temp = ""
        words = bookText.split(" ")[i:n]
        # print ("----->",words)
        for word in words:
            temp = word + " " + temp
        # temp = temp + ',b'
        # print (s)
        # global bBooklist
        # bBooklist.append(temp)
        docs.append(temp)
        labels.append(1)
        i += 150
        n += 150
        x += 1

    return docs, labels


# Cretaing tuple
# cBooklist = []

def readCtxtfile(bookText, docs, labels):
    x = 0
    i = 0
    n = 150
    while x < 200:
        temp = ""
        words = bookText.split(" ")[i:n]
        # print ("----->",words)
        for word in words:
            temp = word + " " + temp
        # temp = temp + ',c'
        # print (s)
        # global cBooklist
        # cBooklist.append(temp)
        docs.append(temp)
        labels.append(2)
        i += 150
        n += 150
        x += 1

    return docs, labels


docs = []
labels = []
docs, labels = readAtxtfile(first_book_text, docs, labels)
# print(aBooklist)
docs, labels = readBtxtfile(second_book_text, docs, labels)
# print(bBooklist)
docs, labels = readCtxtfile(third_book_text, docs, labels)
# print(cBooklist)

print(len(docs))
print(docs)
print(labels)
print(len(labels))

# Data transformation TF-IDF
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 8000, min_df = 3, max_df = 0.6)
TF_X = vectorizer.fit_transform(docs)
TF_X.toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(TF_X, labels, test_size=0.20, random_state=42, shuffle=True)

# fitting the model into machine learning algorithm
# from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import cross_validate
# Training first classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=0.70, gamma='auto',random_state=42)
scores = cross_val_score(clf, x_train, y_train, cv=10)
print(scores)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))
from sklearn.model_selection import StratifiedShuffleSplit
ssf = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
clf = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=21)
#****************End of training of first classifier******************#

#Training Second classifier #
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 42)
scores = cross_val_score(clf,x_train, y_train, cv=10)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))
from sklearn.model_selection import StratifiedShuffleSplit
ssf = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
clf = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 42)
#****************End of training of second classifier******************#

#Training third classifier # k-Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p = 2 )
scores = cross_val_score(clf, x_train, y_train, cv=10)
print("Accuracy: {} (+/- {})".format(scores.mean(), scores.std() * 2))
# manual cross validation with shuffle
from sklearn.model_selection import StratifiedShuffleSplit
ssf = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
clf = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p = 2 )
#****************End of training of Third classifier******************#

new_scores = []
X_array = TF_X.toarray()
labels = np.asarray(labels)
from sklearn.metrics import accuracy_score
for train_index, val_index in ssf.split(X_array, labels):
    x_train, y_train = X_array[train_index], labels[train_index]
    x_val, y_val = X_array[val_index], labels[val_index]
    clf.fit(x_train, y_train)
    prediction_scores = clf.predict(x_val)
    print(accuracy_score(y_val, prediction_scores))
    new_scores.append(accuracy_score(y_val, prediction_scores))

print(np.mean(new_scores))

#*******Confusion Matrix for error analysis/performance analysis

# confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, prediction_scores)
print(cm)

#********End of Confusion Matrix**********************************





'''
cv = cross_validation.KFold(len(train_set), n_folds=10, shuffle=False, random_state=0)
for traincv, testcv in cv:
	classifier = nltk.SklearnClassifier(SVC(kernel = 'rbf', random_state = 0)).train(train_set[traincv[0]:traincv[len(traincv)-1]])
	print ('accuracy:', nltk.classify.util.accuracy(classifier, train_set[testcv[0]:testcv[len(testcv)-1]]))

cv = cross_val_score.KFold(len(train_set), n_folds=10, shuffle=False, random_state=None)
for traincv, testcv in cv:
	classifier = nltk.NaiveBayesClassifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
	print ('accuracy:', nltk.classify.util.accuracy(classifier, train_set[testcv[0]:testcv[len(testcv)-1]]))

'''
