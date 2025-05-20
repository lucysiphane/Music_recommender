#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
music_data=pd.read_csv(r"C:\Users\SIPHANE\Downloads\music.csv")
music_data
#split the data
X=music_data.drop(columns=['genre'])
X
y=music_data['genre']
y
#create model
model=DecisionTreeClassifier()
model.fit(X,y)
predictions=model.predict([[21,1],[22,0]])
predictions
#calculating accuracy of the model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
score=accuracy_score(y_test,predictions)
score



# In[18]:

#visualization of decision tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
music_data=pd.read_csv(r"C:\Users\SIPHANE\Downloads\music.csv")
music_data
X=music_data.drop(columns=['genre'])
X
y=music_data['genre']
y
model=DecisionTreeClassifier()
model.fit(X,y)
tree.export_graphviz(model,out_file='music_recommender.dot',
                     feature_names=['age','gender'],tre
                     class_names=sorted(y.unique()),
                                  label='all',
                                    rounded=True,
                                    filled=True)
                                


tree.plot_tree(model,feature_names=['age','gender'],
               class_names=sorted(y.unique()),
               filled=True)
                        


# In[ ]:




