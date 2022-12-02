#-------------------------------------------------------------------------
# AUTHOR: Kayla Nguyen
# FILENAME: kaylanguyen_assignment5q2.py
# SPECIFICATION: Run k-means multiple times and check which k value maximizes the Silhouette coefficient.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 20 mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:]
y_training = np.array(df.values)[:,-1]
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

maxScore = 0
maxK = 0
for k in range(2,20):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here

    score = silhouette_score(X_training, kmeans.labels_)
    if score > maxScore:
        maxScore = score
        maxK = k

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
print("Max k:", maxK, "\nSilhouette_coefficient:", maxScore)
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1,len(df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
