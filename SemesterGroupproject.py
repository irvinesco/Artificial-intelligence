#import libraries
import matplotlib
import sklearn
import numpy as np
import pandas as pd

#load dataset
dataset = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

#display dataset
dataset.head()

dataset.describe()

dataset.shape

#plot a bar graph
dataset['km_driven'].value_counts().plot(kind='bar', figsize=(20,10))

#classification of datatype
dataset.dtypes

#Label Encoding
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
dataset1 = dataset
for i in dataset1.columns:
    dataset1[i]= lb.fit_transform(dataset1[i])
    
#Display for dataset1
dataset1.head()    
