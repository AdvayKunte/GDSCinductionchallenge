# Importing necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load or train the model with st.cache so that model will not be required to run everytime
@st.cache(allow_output_mutation=True)
def train_model():
    # Importing training data
    dtr = pd.read_csv('Train.csv')

    # Dropping columns of no use to the algorithm
    dtr = dtr.drop(columns=['ID'])

    # Removing the target variable column
    ytr = dtr.pop('Segmentation')

    # Converting categorical data into numerical
    name_col = ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
    dtr = pd.get_dummies(dtr,columns = name_col, drop_first = True)

    # Replacing the NaN values in the NaN columns using the median
    nancols = ['Age', 'Family_Size', 'Work_Experience']
    for item in nancols:
        m_val = dtr[item].median()
        dtr[item].fillna(value=m_val, inplace=True)

    # Encoding the categorization in segmentation
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    ytr = np.array([label_mapping[label] for label in ytr])

    # Scaling data
    s = StandardScaler()
    xtr = s.fit_transform(dtr)

    # DBSCAN for outlier removal
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    outliers = dbscan.fit_predict(xtr)
    indxoutlier = [x for x in range(np.shape(outliers)[0]) if outliers[x] == -1]
    xtr = np.delete(xtr, indxoutlier, axis=0)
    ytr = np.delete(ytr, indxoutlier, axis=0)

    # Splitting the training data into train and evaluation data
    xtr, xev, ytr, yev = train_test_split(xtr, ytr, test_size=0.3, random_state=42)

    # Creating the neural network architecture
    sp = np.shape(xtr)
    lin = tf.keras.layers.InputLayer(input_shape=sp[1:])
    flatten_layer = Flatten()
    layer1 = Dense(units=256, activation='relu')
    layer2 = Dense(units=32, activation='relu')
    layer3 = Dense(units=4, activation='softmax')

    model = Sequential([lin, flatten_layer, layer1, layer2, layer3])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(xtr, ytr, epochs=10)

    #Testing the accuracy of neural network classification model on eval data
    yp = model.predict(xev)
    ypr = []
    for item in yp:
      y0 = 0
      for x in range(4):
        if(item[x]>item[y0]):
          y0 = x
      ypr.append(y0)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(yev,ypr))

##    #Trying using :Logisctic Regression
##    from sklearn.linear_model import LogisticRegression
##    lr = LogisticRegression()
##    lr.fit(xtr,ytr)
##    ypred = lr.predict(xev)
##    print("Accuracy score using Logistic Regression "+ str(accuracy_score(yev,ypred)))
##
##    #Trying using XGBoost Classifier
##    from xgboost import XGBClassifier
##    clm = XGBClassifier()
##    clm.fit(xtr,ytr)
##    ypred = clm.predict(xev)
##    print("Accuracy score using XGBClassifier "+ str(accuracy_score(yev,ypred))

    #Maximum accuracy obtained using Neural network model , hence it is used.

    #Reading testing data and preprocessing it
##    tstd = pd.read_csv('Test.csv')
##    tstd = tstd.dropna()
##    dte = tstd.drop(columns =['ID'])
##    name_col = ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
##    dte = pd.get_dummies(dte,columns = name_col, drop_first = True)
##    st = StandardScaler()
##    xte = st.fit_transform(dte)
##    #
##    ypred = tf.nn.softmax(model.predict(xte))
##    yout = []
##    for item in ypred:
##      y0 = 0
##      for x in range(4):
##        if(item[x]>item[y0]):
##          y0 = x
##      if y0 ==0:
##        yout.append('A')
##      if y0 ==1:
##        yout.append('B')
##      if y0 ==2:
##        yout.append('C')
##      if y0 ==3:
##        yout.append('D')
##    tstd['Prediction']= yout
##    tstd.to_csv('Predicted(nonNaNvalues).csv')
    
    return model

# Load or train the model
trained_model = train_model()

# Streamlit app code
st.title('Customer Segmentation')
st.header("Answer the following questions and find the right customer segmentation")

# User input using Streamlit widgets
gender = st.radio("Select Gender: ", ('Male', 'Female'))
mstat = st.radio('Ever Married: ', ('Yes', 'No'))
age = st.number_input("Please select age")
gstat = st.radio('Graduated?: ', ('Yes', 'No'))
prof = st.selectbox("Profession: ", ['Artist', 'Engineer', 'Healthcare', 'Entertainment', 'Executive', 'Doctor', 'Homemaker', 'Lawyer', 'Marketing'])
wstat = st.number_input('Work Experience', 0, 100)
sstat = st.selectbox("Spending ?: ", ['Low', 'Average', 'High'])
fstat = st.number_input('Family_Size', 0, 50)
var1 = st.selectbox("Var1 ?: ", ['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'])

# Prediction button
if st.button('Predict'):
    # Preprocess user input data
    user_input = pd.DataFrame({
        'Gender': [gender],
        'Ever_Married': [mstat],
        'Age': [age],
        'Graduated': [gstat],
        'Profession': [prof],
        'Work_Experience': [wstat],
        'Spending_Score': [sstat],
        'Family_Size': [fstat],
        'Var_1': [var1]
    })


    # Scale user input data
    dtr = pd.read_csv('Train.csv')
    dtr = dtr.dropna()
    s = StandardScaler()
    dtr = dtr.drop(columns=['ID','Segmentation'])
    #Concating with training data to get the one hot encoding right
    xy = pd.concat([dtr,user_input],ignore_index=True)
    name_col = ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
    xy = pd.get_dummies(xy,columns = name_col, drop_first = True)
    x = xy.to_numpy()
    s.fit(xy)
    x = s.transform(x)

    # Predict using the trained model
    ypred = trained_model.predict(x)

    # Get the index of the maximum probability of the last predicted entry 
    predicted_class_index = np.argmax(ypred[-1])

    # Map the index to the corresponding segmentation label
    label_mapping_reverse = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    predicted_segmentation = label_mapping_reverse[predicted_class_index]

    # Display the predicted segmentation
    st.write("Predicted segmentation:", predicted_segmentation)

