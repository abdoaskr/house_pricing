
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

train = pd.read_csv('I:/ASU/AI/Home project/train.csv')
test = pd.read_csv('I:/ASU/AI/Home project/test.csv')



train.drop(columns=["Id"],inplace=True)
train.drop(columns=train.columns[train.isnull().sum().values>200],inplace=True)
train.dropna(inplace=True)
train.isnull().sum().values


obj_to_replace = train["MSZoning"].dtype

for column in train.columns:
    if train[column].dtype == obj_to_replace:
        uniques = np.unique(train[column].values)
        for idx, item in enumerate(uniques):
            train[column] = train[column].replace(item, idx)

train["bias"] = np.ones(train.shape[0])




train = train.sample(frac=1).reset_index(drop=True)
training_df = train[:-100]
val_df = train[-100:]
training_y = training_df["SalePrice"].values
training_X = training_df.drop(columns=["SalePrice"]).values
val_y = val_df["SalePrice"].values
val_X = val_df.drop(columns=["SalePrice"]).values

# prepare the required data

y_actual = train[['SalePrice']]
input_array = train[['MSSubClass','LotArea','bias']]
input_array2 = train[['MSSubClass','LotArea']]

# Do the calculations
weights = np.dot((np.dot(np.linalg.inv(np.dot(input_array.T,input_array)),input_array.T)),y_actual)


y_pred = np.dot(input_array,weights)


# Measure errors

error = mae(y_pred, y_actual)

# display
print("Mean absolute error : " + str(error))



# BY_USING_SKLEARN_METHOD


X = np.array(input_array2)
y = np.array(y_actual)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# error
error = mae(y_true=y_test,y_pred=y_pred)

print("Mean absolute error with sklearn : " + str(error))



