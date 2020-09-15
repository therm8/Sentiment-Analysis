#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('tweets_GroundTruth.txt', sep='\t', header = None)


# In[3]:


df.head()


# In[4]:


df = df.drop(columns=0)


# In[5]:


df.columns=['rating', 'tweet']


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


df.loc[(df.rating > 0.2),'rating']= 1
df.loc[(df.rating < 0.2) & (df.rating > -0.2),'rating']= 0
df.loc[(df.rating < -0.2),'rating']= -1


# In[9]:


df = df.astype(dtype= {"rating":"int64",
        "tweet":"object"})


# In[10]:


df.dtypes


# In[11]:


df['rating'].value_counts()


# In[12]:


df['tweet'] = df.tweet.map(lambda x: x.lower()) #lowercaseing all tweets


# In[13]:


df['tweet'] = df.tweet.str.replace('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '') #remove URLs


# In[14]:


df['tweet'] = df.tweet.str.replace('@[^\s]+', '') #remove twitter handles


# In[15]:


df['tweet'] = df.tweet.str.replace('[^\w\s]', '') #remove punctuation


# In[16]:


df['tweet'] = df.tweet.str.replace(' +', ' ') #remove some unnecessary whitespaces that showed up when removing other stuff


# In[17]:


stop = stopwords.words('english')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[18]:


df['tweet'] = df['tweet'].apply(nltk.word_tokenize)


# In[19]:


stemmer = PorterStemmer()
df['tweet'] = df['tweet'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[20]:


df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x))


# In[39]:


df.head()


# In[21]:


vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['tweet'])


# In[22]:


xTrain, xTest, yTrain, yTest = train_test_split(x, df['rating'], test_size = 0.2, random_state = 42)


# In[23]:


tree = DecisionTreeClassifier().fit(xTrain, yTrain)
nn = MLPClassifier().fit(xTrain, yTrain)
logreg = LogisticRegression().fit(xTrain, yTrain)
nb = MultinomialNB().fit(xTrain, yTrain)
svm = SVC(kernel='linear').fit(xTrain, yTrain)


# In[24]:


result_tree = tree.predict(xTest)
result_nn = nn.predict(xTest)
result_logreg = logreg.predict(xTest)
result_nb = nb.predict(xTest)
result_svm = svm.predict(xTest)


# In[26]:


print('acc scores')
print('DecTree: ' + str(accuracy_score(yTest, result_tree)))
print('NeuralNet: ' + str(accuracy_score(yTest, result_nn)))
print('LogReg: ' + str(accuracy_score(yTest, result_logreg)))
print('Naive bayes: ' + str(accuracy_score(yTest, result_nb)))
print('SVM: ' + str(accuracy_score(yTest, result_svm)))


# In[27]:


print('Decision Tree')
print(confusion_matrix(yTest, result_tree))
print('Neural Network')
print(confusion_matrix(yTest, result_nn))
print('Logistic Regression')
print(confusion_matrix(yTest, result_logreg))
print('Naive Bayes')
print(confusion_matrix(yTest, result_nb))
print('SVM')
print(confusion_matrix(yTest, result_svm))


# In[28]:


print('Decision Tree')
print(classification_report(yTest, result_tree))
print('Neural Network')
print(classification_report(yTest, result_nn))
print('Logistic Regression')
print(classification_report(yTest, result_logreg))
print('Naive Bayes')
print(classification_report(yTest, result_nb))
print('SVM')
print(classification_report(yTest, result_svm))


# In[29]:


from yellowbrick.classifier import ClassificationReport


# In[30]:


viz_tree = ClassificationReport(DecisionTreeClassifier())
viz_tree.fit(xTrain, yTrain)
viz_tree.score(xTest, yTest)
viz_tree.show()

viz_nn = ClassificationReport(MLPClassifier())
viz_nn.fit(xTrain, yTrain)
viz_nn.score(xTest, yTest)
viz_nn.show()

viz_logreg = ClassificationReport(LogisticRegression())
viz_logreg.fit(xTrain, yTrain)
viz_logreg.score(xTest, yTest)
viz_logreg.show()

viz_nb = ClassificationReport(MultinomialNB())
viz_nb.fit(xTrain, yTrain)
viz_nb.score(xTest, yTest)
viz_nb.show()

viz_svm = ClassificationReport(SVC(kernel='linear'))
viz_svm.fit(xTrain, yTrain)
viz_svm.score(xTest, yTest)
viz_svm.show()


# In[30]:


df2 = pd.read_csv('fpl_tweets.csv')


# In[31]:


df2


# In[32]:


df2 = df2.drop(columns=['date', 'username', 'to', 'replies', 'retweets', 'favorites', 'geo', 'mentions', 'hashtags', 'id', 'permalink'])


# In[34]:


df2


# In[35]:


df2['text'] = df2.text.map(lambda x: x.lower()) #lowercaseing all tweets
df2['text'] = df2.text.str.replace('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '') #remove URLs
df2['text'] = df2.text.str.replace('@[^\s]+', '') #remove twitter handles
df2['text'] = df2.text.str.replace('[^\w\s]', '') #remove punctuation
df2['text'] = df2.text.str.replace(' +', ' ') #remove some unnecessary whitespaces
df2['text'] = df2['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df2['text'] = df2['text'].apply(nltk.word_tokenize)
df2['text'] = df2['text'].apply(lambda x: [stemmer.stem(y) for y in x])
df2['text'] = df2['text'].apply(lambda x: ' '.join(x))


# In[36]:


df2


# In[37]:


fpl_vect = vectorizer.transform(df2['text'])
evaluate_tree = tree.predict(fpl_vect)
evaluate_nn = nn.predict(fpl_vect)
evaluate_logreg = logreg.predict(fpl_vect)
evaluate_nb = nb.predict(fpl_vect)
evaluate_svm = svm.predict(fpl_vect)


# In[38]:


pos_tree = 0
neg_tree = 0
neu_tree = 0
for i in evaluate_tree:
    if i == 1:
        pos_tree += 1
    elif i == 0:
        neu_tree += 1
    else:
        neg_tree += 1
print('Decision Tree')
print('Positive: ' + str(pos_tree))
print('Neutral: ' + str(neu_tree))
print('Negative: ' + str(neg_tree))
print(' ')

pos_nn = 0
neg_nn = 0
neu_nn = 0
for i in evaluate_nn:
    if i == 1:
        pos_nn += 1
    elif i == 0:
        neu_nn += 1
    else:
        neg_nn += 1
print('Neural Net')
print('Positive: ' + str(pos_nn))
print('Neutral: ' + str(neu_nn))
print('Negative: ' + str(neg_nn))
print(' ')

pos_logreg = 0
neg_logreg = 0
neu_logreg = 0
for i in evaluate_logreg:
    if i == 1:
        pos_logreg += 1
    elif i == 0:
        neu_logreg += 1
    else:
        neg_logreg += 1
print('Logistic Regression')
print('Positive: ' + str(pos_logreg))
print('Neutral: ' + str(neu_logreg))
print('Negative: ' + str(neg_logreg))
print(' ')

pos_nb = 0
neg_nb = 0
neu_nb = 0
for i in evaluate_nb:
    if i == 1:
        pos_nb += 1
    elif i == 0:
        neu_nb += 1
    else:
        neg_nb += 1
print('Naive Bayes')
print('Positive: ' + str(pos_nb))
print('Neutral: ' + str(neu_nb))
print('Negative: ' + str(neg_nb))
print(' ')

pos_svm = 0
neg_svm = 0
neu_svm = 0
for i in evaluate_svm:
    if i == 1:
        pos_svm += 1
    elif i == 0:
        neu_svm += 1
    else:
        neg_svm += 1
print('SVM')
print('Positive: ' + str(pos_svm))
print('Neutral: ' + str(neu_svm))
print('Negative: ' + str(neg_svm))
print(' ')

