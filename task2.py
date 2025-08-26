import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
df = sns.load_dataset('titanic')


df.head()

df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who'], axis=1)


df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)


le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  
df['embarked'] = le.fit_transform(df['embarked'])  


df.dropna(inplace=True)


X = df.drop('survived', axis=1) 
y = df['survived']              


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
