# Creating a deep learning regression model
#Making necessary imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten

#Importing data
dtr = pd.read_csv('training_data_student_perf.csv')
#Removing target variable from training data
ytr = dtr.pop('Performance Index')
#Categorical columns encoded using numerical values
dtr['Extracurricular Activities']=dtr['Extracurricular Activities'].astype('category')
dtr['Extracurricular Activities']=dtr['Extracurricular Activities'].cat.codes
#Dropping NaN values
dtr = dtr.dropna()
#Preprocessing the Standard Scaler
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
xtr = s.fit_transform(dtr)
#Splitting the data into training and testing split
from sklearn.model_selection import train_test_split
xtr , xev,ytr,yev = train_test_split(xtr,ytr,test_size=0.3,random_state=42)

#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtr,ytr)
ypr = lr.predict(xev)
from sklearn.metrics import r2_score
#print(r2_score(ypr,yev))

# Neural Network
sp = np.shape(xtr)
lin = tf.keras.layers.InputLayer(input_shape=sp[1:])
flatten_layer = Flatten()
layer1 = Dense(units=128, activation='relu')
layer2 = Dense(units=16, activation='linear')
layer3 = Dense(units = 1, activation = 'linear')

model = Sequential([lin, flatten_layer,layer1,layer2,layer3])
model.summary()
model.compile(loss='mean_squared_error')

model.fit(xtr,ytr)
ypr = model.predict(xev)
#print(r2_score(yev,ypr))

#Regressor using Extreme Gradient Boost
from xgboost import XGBRegressor
clm = XGBRegressor(objective ='reg:linear')
clm.fit(xtr,ytr)
ypr = clm.predict(xev)
#print(r2_score(ypr,yev))

#Predicting Testing Data Values

#Preprocessing testing data
dte = pd.read_csv('test_data_student_perf.csv')
dte = dte.drop(columns=['ID'])
dte['Extracurricular Activities']=dte['Extracurricular Activities'].astype('category')
dte['Extracurricular Activities']=dte['Extracurricular Activities'].cat.codes
dte = dte.dropna()
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
xte = st.fit_transform(dte)
yout = lr.predict(xte)
dte['Predicted']=yout
dte.to_csv('Prediction.csv')

import streamlit as st
# Streamlit app code
st.title('Performance Prediciton')
st.header("Answer the following questions and find the performance of the student")

# User input using Streamlit widgets
h = st.number_input("Please enter hours studied")
ps = st.number_input("Please enter previous scores")
ex = st.radio('Extracurricular Activities?: ',('Yes','No'))
sh = st.number_input("Please enter sleep hours")
qp = st.number_input("Please enter sample question papers practiced")

# Prediction button
if st.button('Predict'):
    # Preprocess user input data
    x = pd.DataFrame({
    'Hours Studied': [h],
    'Previous Scores': [ps],
    'Extracurricular Activities': [ex],
    'Sleep Hours': [sh],
    'Sample Question Papers Practiced ': [qp]
    })
    item = 'Extracurricular Activities'
    x[item] = x[item].astype('category')
    x[item] = x[item].cat.codes

    xp =x.to_numpy()
    xp = s.transform(xp)
    xp = xp.reshape(1,-1)
    yp = lr.predict(xp)
    st.write("Predicted Performance Index -: " + str(yp))
