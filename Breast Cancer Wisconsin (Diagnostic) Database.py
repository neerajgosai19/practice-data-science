#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[5]:


cancer.keys()


# In[13]:


def answer_zero():
     return len(cancer['feature_names'])

answer_zero() 


# In[12]:


print(cancer.DESCR)


# In[17]:


def answer_one():
    
    columns = np.append(cancer.feature_names, 'target');
    print("Feature column size::: " + str(np.size(columns)))
    index = pd.RangeIndex(start=0 , stop=569, step=1);
    # Append target data to current data
    data = np.column_stack((cancer.data , cancer.target))
    print("Data column Size:::" + str(np.size(data)/569))
    # Create dataframe with keywords
    df = pd.DataFrame(data=data , index=index, columns=columns)
    return df
answer_one()

     
   


# In[19]:


def answer_two():
    cancerdf = answer_one()
    index =['malignant', 'benign']
    malignant = np.where(cancerdf['target'] == 0.0);
    benign = np.where(cancerdf['target'] == 1.0);
    # Get sizes and build
    data = [np.size(malignant) , np.size(benign)]
    print(data)
    # Construct a series object
    series = pd.Series(data, index=index)    
    
    return series # Return your answer

answer_two()


# In[20]:


def answer_three():
    cancerdf = answer_one()
    X = cancerdf.drop('target', axis=1)
    y = cancerdf.get('target')
    
    
    return X, y


# In[21]:


#Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
#Set the random number generator state to 0 using random_state=0
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test =  train_test_split(X, y , random_state = 0 )
        
    return X_train, X_test, y_train, y_test


# In[27]:


#Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).
from sklearn.neighbors import KNeighborsClassifier
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    knn.score(X_test ,y_test)  
    return knn
answer_five()


# In[28]:


def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()
    return knn.predict(means)
answer_six()


# In[29]:


#Using your knn classifier, predict the class labels for the test set X_test
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    test_prediction = knn.predict(X_test)   
    return test_prediction
answer_seven()


# In[30]:


#Find the score (mean accuracy) of your knn classifier using X_test and y_test.
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = knn.score( X_test, y_test)
    print('Accuracy :: = ' + str(score))
    prediction = answer_six()
    print("Going to be cancer: " + str())  
    return score
answer_eight()


# In[35]:


def accuracy_plot():
    import matplotlib.pyplot as plt

    get_ipython().run_line_magic('matplotlib', 'notebook')

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




