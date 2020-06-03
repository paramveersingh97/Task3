#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
import os
dataset=mnist.load_data("mymnist.db")


# In[2]:


len(dataset)


# In[3]:


train,test=dataset


# In[4]:


x_train,y_train=train


# In[5]:


x_test,y_test=test
len(x_test)


# In[6]:


x_train_vector=x_train.reshape(-1,28*28)  #conertin in to 1d array,numeric vector


# In[7]:


from keras.utils.np_utils import to_categorical


# In[8]:


y_train_categories=to_categorical(y_train)


# In[9]:


x_test_vector=x_test.reshape(-1,28*28)


# In[10]:


y_test_categories=to_categorical(y_test)


# In[11]:


from keras.models import Sequential
from keras.layers import Dense


# In[12]:


model=Sequential()
model.add(Dense(units=512,input_shape=(28*28,),kernel_initializer='glorot_uniform',activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))


# In[13]:


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[14]:


history=model.fit(x_train_vector,y_train_categories,epochs=1)


# In[15]:


correct=model.evaluate(x_test_vector,y_test_categories)
acc=correct[1]*100


# In[16]:


f = open("accuracy.txt", "w")
f.write(str(acc))
f.close()

#open and read the file after the appending:
f = open("accuracy.txt", "r")
print(f.read())


# In[17]:


model.save("task3.h5")

