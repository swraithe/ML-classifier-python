import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(<pathtofile>)

X=dataset.iloc[<range of rows and input columns>].values y=dataset.iloc[<range of rows and output column>].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(X[<range of rows and columns>])
X[<range of rows and columns>]=imputer.transform(X[<range of rows and columns>])

from sklearn.preprocessing import LabelEncoder
le_X=LabelEncoder()
X[<range of rows and columns>]=le_X.fit_transform(X[<range of rows and columns>])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
oneHE=OneHotEncoder(categorical_features=[<range of rows and columns>])
X=oneHE.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

from sklearn.<class module> import <model class>
classifier = <model class>(<parameters>)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
 
from sklearn.metrics import classification_report
target_names = [<list of class names>]
print(classification_report(y_test, y_pred, target_names=target_names))

