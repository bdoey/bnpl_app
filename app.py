import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Loan Prediction Web App')
st.write('This is a web app to predict the risk of a loan.')

                        
loan_amnt = st.slider('Loan Amount', 500, 100000, 1000)
term = st.select_slider('Term', options=[36, 60])
emp_length = st.slider('Employment Length', 0, 10, 1)
annual_inc = st.slider('Annual Income', 10000, 300000, 1000)
dti = st.slider('Debt-to-Income Ratio', 0, 50, 1),
delinq_2yrs = st.slider('Delinquencies', 0, 10, 1),
open_acc = st.slider('Open Accounts', 0, 50, 1),
mort_acc = st.slider('Mortage Accounts', 0, 50, 1), 
pub_rec_bankruptcies = st.slider('Bankruptcies', 0, 10, 1),
last_fico_range_high = st.slider('FICO Score', 300, 850, 10),
earliest_cr_line = st.slider('Earliest Credit Line', 1990, 2020, 1)


features = {'Loan Amount':loan_amnt,
            'Term':term,
            'Employment Length':emp_length,
            'Annual Income':annual_inc,
            'Debt-to-Income Ratio':dti,
            'Delinquencies':delinq_2yrs,
            'Open Accounts':open_acc,
            'Mortage Accounts':mort_acc,
            'Bankruptcies':pub_rec_bankruptcies,
            'FICO Score':last_fico_range_high,
            'Earliest Credit Line':earliest_cr_line
            }


features_df  = pd.DataFrame([features])

st.table(features_df)  


if st.button('Predict'):
        result = predict(np.array([loan_amnt, term, emp_length, annual_inc, dti, delinq_2yrs, open_acc, mort_acc, pub_rec_bankruptcies, last_fico_range_high]))
        st.text(result[0])
