#import libraries
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd

#load dataset
dataset = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

#display dataset
dataset.head()

dataset.describe()
#Data cleaning and preprocessing
dataset.isna().sum()

dataset.shape

dataset.columns

#Check for value counts
dataset['owner'].value_counts()

#Check no of cars going for less than 250000
dataset[dataset['selling_price']<250000].count()['selling_price']
#Display the dataset
dataset[dataset['selling_price']<250000]

#Check no of cars going for more than 250000
dataset[dataset['selling_price']>250000].count()['selling_price']
#Display the dataset
dataset[dataset['selling_price']>250000]

#plot a bar graph
dataset['owner'].value_counts().plot(kind='bar', figsize=(20,10))

import seaborn as sns
sns.countplot(dataset['owner'])
sns.displot(dataset['owner'])

#classification of datatype
dataset.dtypes

#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
dataset1 = dataset
for i in dataset1.columns:
    dataset1[i]= lb.fit_transform(dataset1[i])
    
#Display for dataset1
dataset1.head()    

dataset.describe()
#classification of datatype1
dataset1.dtypes
#Dataset information
dataset1.info()

dataset1.shape


