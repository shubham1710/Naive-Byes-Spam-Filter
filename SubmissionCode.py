#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
a=pd.read_csv("train_X.csv", encoding='latin-1', header=None)
a1=pd.read_csv("test_X_data.csv", encoding='latin-1', header=None)
b=pd.read_csv("train_Y.csv", header=None)
a.columns = ["text"]
a1.columns = ["text"]
b.columns = ["value"]
x=a.text
x1=a1.text
y=b.value
vect = CountVectorizer()
counts = vect.fit_transform(x.values)
counts1=vect.transform(x1.values)
classifier = MultinomialNB()
targets = y.values
classifier.fit(counts,targets)
predictions = classifier.predict(counts1)
co=np.shape(x)[0]
predictions.resize(co,1)
np.savetxt("predicted_test_Y.csv", predictions, delimiter=",")
print(accuracy_score(y,predictions))


# In[ ]:




