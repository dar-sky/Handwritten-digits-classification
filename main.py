#!/usr/bin/env python
# coding: utf-8
#  auth :: Pradeep
# In[1]:


import numpy as np
import seaborn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow 
from tensorflow import keras


# In[2]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


plt.matshow(X_train[101])


# In[4]:


y_train[101]


# In[5]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[6]:




# In[7]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='sigmoid'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20,activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=10)


# In[8]:


model.evaluate(X_test_flattened,y_test)


# In[9]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tensorflow.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
seaborn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')








