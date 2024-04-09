# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:37:34 2023

@author: Krushna Lad
"""
"""

Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. 
After that, perform PCA on the dataset and extract the first 
3 principal components and make a new dataset with these 3 
principal components as the columns. Now, on this new dataset, 
perform hierarchical and K-means clustering. Compare the results 
of clustering on the original dataset and clustering on the 
principal components dataset (use the scree plot technique to
obtain the optimum number of clusters in K-means clustering and
check if you’re getting similar results with and without PCA).

"""

"""
Business Objectives
Maximize:- 
Minimize:-
Business Constraint:-

"""
"""
Data Dictionary

Name of features          Type     Relevance      Description
0              Type       Nominal  Relevant  Type of alcohol
1           Alcohol    Continuous  Relevant         Alcohol 
2             Malic    Continuous  Relevant            Malic
3               Ash    Continuous  Relevant              Ash
4        Alcalinity    Continuous  Relevant       Alcalinity
5         Magnesium  Quantitative  Relevant        Magnesium
6           Phenols    Continuous  Relevant          Phenols
7        Flavanoids    Continuous  Relevant       Flavanoids
8     Nonflavanoids    Continuous  Relevant    Nonflavanoids
9   Proanthocyanins    Continuous  Relevant  Proanthocyanins
10            Color    Continuous  Relevant            Color
11              Hue    Continuous  Relevant              Hue
12         Dilution    Continuous  Relevant         Dilution
13          Proline  Quantitative  Relevant          Proline

"""
##EDA (Exploratory Data Analysis)##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Dataset/wine.csv")
print(df)
print(df.head())

## 5-number Summary ##
df.describe()
#[8 rows x 14 columns]
df.shape
#(rows=178, columns=14)
df.columns

#Value counts
df['Type'].value_counts()
"""
2    71
1    59
3    48
"""
#Check for null values
df.isnull
#False

df.isnull().sum()
# There is no any null value present in the dataset

#Scatter Plot
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Type") \
   .map(plt.scatter, "Alcohol", "Phenols") \
   .add_legend();
plt.show();
# Blue points is Type 1 , orange is Type 2 and Green
# is Type 3  
# But red and green data points cannot be easily seperated.

# displot for Alcohol on Type
sns.FacetGrid(df, hue="Type") \
   .map(sns.distplot, "Alcohol") \
   .add_legend();
plt.show();

# displot for color on Type
sns.FacetGrid(df, hue="Type") \
   .map(sns.distplot, "Color") \
   .add_legend();
plt.show();

# Pairwise scatter plot: Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Type");
plt.show()

##Boxplot##
#Take a boxplot for Alcohol column
sns.boxplot(df.Alcohol)
# In alcohol column no outliers 

#Now Take a boxplot on df column
sns.boxplot(df)
# There are outliers in some of the columns

#Let's take a boxplot for Malic column
sns.boxplot(df.Malic)
#There are Three outliers in malic Column

#Take a boxplot for df column
sns.boxplot(df.Ash)
#Here also 3 outliers present in the column

##Histplot##
sns.histplot(df['Alcohol'],kde=True)
# data is right-skewed and it is not normallly distributed
# Data is right-skewed and not symmetric

sns.histplot(df['Ash'],kde=True)
#Here data is right-skewed and not normallly distributed
#Data is not symmetric

sns.histplot(df,kde=True)
#The data is showing the skewness 
#Maximum of data is right skewed 

##Data Preproccesing##
df.dtypes
# Type and Proline is in int and other all columns are 
#in float data types

# Now Identify the duplicates present in Dataset #

duplicate=df.duplicated()
# Output of this function is in single column
# if there is duplicate records then the output will be - True
# if there is no duplicate records then the output will be - False
# Series is created
duplicate
# All Are showing Output as False
#It means there is no any duplicate records are present.
sum(duplicate)
# output is zero

# As We found outliers in some of the columns 
#Hence we need here some outlier treatments

##Outliers treatment##

# Winsorizer #
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Malic']
                  )
df_t=winsor.fit_transform(df[['Malic']])

sns.boxplot(df[['Malic']])
# There are outliers present in data
# check after applying the Winsorizer
sns.boxplot(df_t['Malic'])
#Now Outliers are removed

# Label encoder
# preferaly for nominal data

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# creating instance of label
labelencoder=LabelEncoder()
# split your data into input and output variables
x=df.iloc[:,0:]
y=df['Type']
df.columns

# we have nominal data Type
# we want to convert to label encoder
x['Type']=labelencoder.fit_transform(x['Type'])
# label encoder y
y=labelencoder.fit_transform(y)
# This is going to create an array, hence convert
# It is back to dataframe
y=pd.DataFrame(y)
df_new=pd.concat([x,y],axis=1)
# If you will see variables explorer, y do not have column name

# hence the rename column
df_new=df_new.rename(columns={0:'Typ'})

# Normalization

# Normalization function
# whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(df) 
# you can check the df_norm dataframe which is scaled between values from 0 and 1
b=df_norm.describe()
# Data is normalize
# in 0-1 

# Before we can apply clustering , need to plot dendrogram first
# now to create dendrogram , we need to measure distance,

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
# Linkage function gives us hierarchical clustering
z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
# ref help of dendrogram
# sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
# dendrongram()
# applying agglomerative clustering choosing 4 as clustrers
# from dendrongram
# whatever has been displayed in dendrogram is not clustering
# It is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

# Assign this series to df Dataframe as column and name the column
df['clust']=cluster_labels
# we want to restore the column 7th position to 0th position
df1=df.iloc[:,:]
# now check the df dataframe

df1.to_csv("first.csv",encoding="utf-8")
import os
os.getcwd()

# K-Means
from  sklearn.cluster import KMeans

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)# total within sum of square


TWSS
# As k value increases the TWSS the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_clusters")

