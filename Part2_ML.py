# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

------------------------------Part2: Machine Learning----------------------------------

--------------------------@author: weaam almutawwa, 438201478--------------------------


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#======================================================================================#
#                                 Import libraries                                     #
#======================================================================================#

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#======================================================================================#
#                                      stopwords                                       #
#======================================================================================#

nltk.download('stopwords')

#======================================================================================#
#                                      get data                                        #
#======================================================================================#

def get_data(faketxt, realtxt):
    
    # read files and convert to data frame
    fake = pd.read_table(faketxt)
    real = pd.read_table(realtxt)
    
    #fake splitting
    ind_ftrain = int(round(fake.shape[0]*0.7))
    ind_ftest = int(round(fake.shape[0]*(0.7+0.15)))
    fake_train = fake[:ind_ftrain]
    fake_test = fake[ind_ftrain:ind_ftest]
    fake_val = fake[ind_ftest:]
    
    #real splitting
    ind_rtrain = int(round(real.shape[0]*0.7))
    ind_rtest = int(round(real.shape[0]*(0.7+0.15)))
    real_train = real[:ind_rtrain]
    real_test = real[ind_rtrain:ind_rtest]
    real_val = real[ind_rtest:]

    return fake, real, fake_train, fake_test, fake_val, real_train, real_test, real_val

#======================================================================================#
#                                   get features                                       #
#======================================================================================#

def get_features(faketxt, realtxt):
    
    fake = open(faketxt, "r")
    real = open(realtxt, "r")

    stop_words = stopwords.words('english')
    d = dict()
    features = []
    
    # features from fake headlines
    for line in fake:
        line = line.strip() 
        line = line.lower()
        words = line.split(" ")
        for word in words:
            if word in d:
                d[word] = d[word] + 1
            else: 
                d[word] = 1
                
    # features from real headlines
    for line in real:
        line = line.strip() 
        line = line.lower()
        words = line.split(" ")
        for word in words:
            if not word in stop_words:
                if word in d:
                    d[word] = d[word] + 1
                else: 
                    d[word] = 1
                
    orderedset = OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse = True))
        
    for key in list(orderedset.keys()):
        if (d[key] > 5 ):
            features.append(key)
    return features

#======================================================================================#
#                               make metrix and label                                  #
#======================================================================================#

def make_matrix(fake, real):
    
    x = np.concatenate([real, fake])
    x_label = ['real']*real.shape[0] + ['fake']*fake.shape[0]
    
    return x, x_label

#======================================================================================#
#                          convert numeric to use by model                             #
#======================================================================================#

def convertNumeric(data): 
    
    vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 835) 
    conv_data = vectorizer.fit_transform(data.ravel())
    
    return conv_data

#======================================================================================#
#                                   decision tree                                      #
#======================================================================================#

def select_modelDT(train, train_label, test, test_label, val, val_label, features):
    
    print("----------------------------------")
    print("------Decision Tree Training------")
    
    # 1st Model
    classifier1 = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    classifier1.fit(train, train_label)
    y_pred = classifier1.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('1st Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 2nd Model
    classifier2 = DecisionTreeClassifier(criterion="gini", max_depth=5)
    classifier2.fit(train, train_label)
    y_pred = classifier2.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('2nd Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 3st Model
    classifier3 = DecisionTreeClassifier(criterion="entropy", max_depth=12)
    classifier3.fit(train, train_label)
    y_pred = classifier3.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('3rd Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 4th Model
    classifier4 = DecisionTreeClassifier(criterion="gini", max_depth=20)
    classifier4.fit(train, train_label)
    y_pred = classifier4.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('4th Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))

    # 5th Model
    classifier5 = DecisionTreeClassifier(criterion="entropy", max_depth=30)
    classifier5.fit(train, train_label)
    y_pred = classifier5.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('5th Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
     
    # Testing
    y_pred = classifier5.predict(test)
    accuracy = 100*accuracy_score(test_label, y_pred)
    print('Testing Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(test_label, y_pred))
    print(classification_report(test_label, y_pred))
    
    # Visualizing the tree
    fig = plt.figure(figsize=(20,20))
    _ = tree.plot_tree(classifier5, max_depth = 2, feature_names=features, filled=True)
        
    
#======================================================================================#
#                                logistic Regression                                   #
#======================================================================================#

def select_modelLR(train, train_label, test, test_label, val, val_label):

    print("----------------------------------")
    print("---Logistic Regression Training---")
    
    # 1st Model
    classifier1 = LogisticRegression(solver="liblinear")
    classifier1.fit(train, train_label)
    y_pred = classifier1.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('1st Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 2nd Model
    classifier2 = LogisticRegression(solver="saga")
    classifier2.fit(train, train_label)
    y_pred = classifier2.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('2nd Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 3rd Model    
    classifier3 = LogisticRegression(solver="newton-cg")
    classifier3.fit(train, train_label)
    y_pred = classifier3.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('3rd Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 4th Model
    classifier4 = LogisticRegression(solver="lbfgs")
    classifier4.fit(train, train_label)
    y_pred = classifier4.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('4th Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # 5th Model
    classifier5 = LogisticRegression(solver="sag")
    classifier5.fit(train, train_label)
    y_pred = classifier5.predict(val)
    accuracy = 100*accuracy_score(val_label, y_pred)
    print('5th Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(val_label, y_pred))
    print(classification_report(val_label, y_pred))
    
    # Testing
    y_pred = classifier1.predict(test)
    accuracy = 100*accuracy_score(test_label, y_pred)
    print('Testing Model Accuracy: %f %%' %accuracy)
    print(confusion_matrix(test_label, y_pred))
    print(classification_report(test_label, y_pred))
    
#=========================================================================================================#
#----------------------------------------------------------------------------------------------------------
# file names
faketxt = 'clean_fake.txt'
realtxt = 'clean_real.txt'

#----------------------------------------------------------------------------------------------------------
# data preperation
fakedata, realdata, faketrain, faketest, fakeval, realtrain, realtest, realval = get_data(faketxt, realtxt)

#----------------------------------------------------------------------------------------------------------
# adding labels
fakedata['label'] = 'fake'
realdata['label'] = 'real'

#----------------------------------------------------------------------------------------------------------
# features selection
features = get_features(faketxt, realtxt)

#----------------------------------------------------------------------------------------------------------
# labeling sets
train_x, train_y = make_matrix(faketrain, realtrain)
test_x, test_y = make_matrix(faketest, realtest)
val_x, val_y = make_matrix(fakeval, realval)

#----------------------------------------------------------------------------------------------------------
# convert to numeric to use by model
train_x = convertNumeric(train_x)
val_x = convertNumeric(val_x)
test_x = convertNumeric(test_x)

#----------------------------------------------------------------------------------------------------------
# shuffling data for sufficient training
train_x, train_y = shuffle(train_x, train_y)
val_x, val_y = shuffle(val_x, val_y)
test_x, test_y = shuffle(test_x, test_y)

#----------------------------------------------------------------------------------------------------------
# decision tree modeling
select_modelDT(train_x, train_y, test_x, test_y, val_x, val_y, features)

#----------------------------------------------------------------------------------------------------------
# logistic regression modeling
select_modelLR(train_x, train_y, test_x, test_y, val_x, val_y)