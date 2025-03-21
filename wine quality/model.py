import pandas as pd
import numpy as np
import pickle

# import dataset
data=pd.read_csv('winequality-red.csv')

#adding purchased column and removing quality column
data['purchased']=data['quality'].apply(lambda x:1 if x>=6 else 0)
data=data.drop('quality',axis=1)

#remove duplicate values
data = data.drop_duplicates()

# spliting the data into x and y
x=data.drop('purchased',axis=1)
y=data['purchased']




#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)

# Save model
pickle.dump(model,open('model.pkl','wb'))