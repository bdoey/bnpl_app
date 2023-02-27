import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import joblib

df = pd.read_csv('/Users/dev1/Python/midterm2/loan_data_clean.csv')

# turn all columns to int type
loan_data = df.astype(int)

# drop loan_status from loan_data and assign to X
X = loan_data.drop('loan_status', axis=1)
y = loan_data['loan_status']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# import XGBClassifier
xgb_model = XGBClassifier()

# Fit the model on training data and make predictions on test data
xgb_model.fit(X_train, y_train)

# make predictions on the testing set
y_pred_class = xgb_model.predict(X_test)

# calculate accuracy and print the result in a nice format
accuracy = metrics.accuracy_score(y_test, y_pred_class)

# print accuracy in % format with 2 decimal places
print('Model Accuracy: {:.2%}'.format(accuracy))

# save the model to disk
joblib.dump(xgb_model, 'xgb_model.sav')
