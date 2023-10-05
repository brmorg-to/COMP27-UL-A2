#!/usr/bin/env python
# coding: utf-8

# # K-Mean & DBSCAN
# - Bruno Morgado (301154898)

# In[1]:


# Necessary Imports
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import defaultdict
import sklearn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split, cross_val_score
import warnings


# In[2]:


# Remove annoying alerts 
warnings.filterwarnings('ignore')


# In[3]:


# Fetch Olivetti dataset from Sklearn
dataset = fetch_olivetti_faces(shuffle=True, random_state=98)


# In[4]:


type(dataset)


# In[5]:


# Storing features, target variable, and 2d features matrix as images
X = dataset.data
y = dataset.target
images = dataset.images  


# In[6]:


type(images)


# In[7]:


images.shape


# In[8]:


# Bundle X and y into a dataframe
pixel_columns = [f"pixel_{i}" for i in range(1, X.shape[1] + 1)]

df = pd.DataFrame(X, columns=pixel_columns)

df['target'] = y


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


# Define a function to plot (default = 40) sample images
def plot_gallery(images, titles, h, w, n_row=5, n_col=8):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# In[12]:


plot_gallery(images, y, h=64, w=64)
plt.show()


# In[13]:


# Split dataset into train, validation, and test sets with stratification
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=98, stratify=y)

X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=98, stratify=y_temp)

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_valid)}")
print(f"Test set size: {len(y_test)}")


# In[14]:


# Instatiate a Support Vector Classifier
svm_clf = SVC(kernel='rbf', random_state=98)


# In[15]:


# Get 5-fold cross validation scores
k = 5
scores = cross_val_score(svm_clf, X_train, y_train, cv=k, scoring='accuracy')

print(f"Cross-validation scores (k={k}):", scores)
print("Average cross-validation score:", scores.mean())


# In[16]:


# Train the SVC classifier
svm_clf.fit(X_train, y_train)


# In[17]:


# Make predictions and print validation scores on the validation set
y_pred_valid = svm_clf.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred_valid)
print(f"Validation accuracy with kernel= rbf:", accuracy)


# In[18]:


# Make predictions on the test set
y_pred = svm_clf.predict(X_test)


# In[19]:


# Print the classification report
print('\t\tClassification Report - SVC\n\n', classification_report(y_test, y_pred))


# In[20]:


# Explore the target variables further
np.unique(y, return_counts=True)


# In[53]:


scores = []
range_clusters = range(2, 200)


# In[54]:


# Fit kmeans with a range of clusters an make predictions
for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init='auto', random_state=98)
    kmeans.fit(X_train)
    preds = kmeans.predict(X_train)
    score = silhouette_score(X_train, preds)
    scores.append(score)


# In[55]:


# Get the silhouette score for each iteration in the for loop above
for i, value in enumerate(scores):
    print(f'Index {i} : {value}')


# In[56]:


# Get the highest silhouette score
best_n_clusters = range_clusters[scores.index(max(scores))]


# In[57]:


best_n_clusters


# In[58]:


best_n_scores = scores[scores.index(max(scores))]


# In[27]:


best_n_scores


# In[59]:


scores[77]


# In[60]:


plt.figure(figsize=(24, 8))
plt.plot(range_clusters, scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.5, 120, 0.05, 0.5])
plt.show()


# In[61]:


# Reduce the dataset dimensionality according to the number of clusters that returned the highest silhouette score
kmeans = KMeans(n_clusters=79, init="k-means++", n_init='auto', random_state=98)


# In[62]:


X_reduced = kmeans.fit_transform(X)


# In[63]:


X_reduced.shape


# In[64]:


# Array with the first instance's distances to the centroids
X_reduced[0]


# In[65]:


# Split the reduced Olivetti dataset 
X_train_reduced, X_temp_reduced, y_train, y_temp = train_test_split(X_reduced, y, test_size=0.3, random_state=98, stratify=y)

X_valid_reduced, X_test_reduced, y_valid, y_test = train_test_split(X_temp_reduced, y_temp, test_size=0.5, random_state=98, stratify=y_temp)

print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_valid)}")
print(f"Test set size: {len(y_test)}")


# In[66]:


# Retrain the classifier with the reduced dataset
svm_clf.fit(X_train_reduced, y_train)


# In[67]:


k = 5
scores = cross_val_score(svm_clf, X_train_reduced, y_train, cv=k, scoring='accuracy')

print(f"Cross-validation scores (k={k}):", scores)
print("Average cross-validation score:", scores.mean())


# In[68]:


y_pred_valid_reduced = svm_clf.predict(X_valid_reduced)
accuracy = accuracy_score(y_valid, y_pred_valid_reduced)
print(f"Validation accuracy with kernel= rbf:", accuracy)


# In[69]:


y_pred_reduced = svm_clf.predict(X_test_reduced)


# In[70]:


# Print the classification report
print('\t\tClassification Report - SVC\n\n', classification_report(y_test, y_pred_reduced))


# ## DBSCAN

# In[71]:


X.shape


# In[72]:


X[:10]


# In[74]:


# Testing the distance and min_samples manually
dbscan = DBSCAN(eps = 7.24, min_samples=2)
clusters = dbscan.fit_predict(X)


# In[75]:


print(np.unique(clusters))


# In[76]:


print(len(np.unique(clusters)))


# In[77]:


#Outliers
print(len(clusters[clusters == -1]))


# In[78]:


images = X.reshape(-1, 64, 64)


# In[79]:


images.shape


# In[80]:


clustered_images = defaultdict(list)
for i, cluster in enumerate(clusters):
    clustered_images[cluster].append(images[i])


# In[81]:


def display_images(images, title=""):
    n_images = len(images)
    rows = int(n_images**0.5)
    cols = (n_images // rows) + (n_images % rows)
    
    plt.figure(figsize=(1.5*cols, 1.5*rows))
    for i in range(n_images):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        ax.axis('off')
    
    plt.suptitle(title)
    plt.show()


# In[50]:


for cluster, images in clustered_images.items():
    display_images(images, title=f"Cluster {cluster}")


# 
