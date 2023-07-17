# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
#import catboost
#from catboost import CatBoostClassifier
import pickle
# load the saved model
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

def xgb_shap_transform_scale(original_shap_values, Y_pred, which):    
    from scipy.special import expit    
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    from scipy.special import expit 
    #Importing the logit function for the base value transformation    
    untransformed_base_value = original_shap_values.base_values[-1]    
    #Computing the original_explanation_distance to construct the distance_coefficient later on    
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]    
    base_value = expit(untransformed_base_value) 
    # = 1 / (1+ np.exp(-untransformed_base_value))    
    #Computing the distance between the model_prediction and the transformed base_value    
    distance_to_explain = Y_pred[which] - base_value    
    #The distance_coefficient is the ratio between both distances which will be used later on    
    distance_coefficient = original_explanation_distance / distance_to_explain    
    #Transforming the original shapley values to the new scale    
    shap_values_transformed = original_shap_values / distance_coefficient    
    #Finally resetting the base_value as it does not need to be transformed    
    shap_values_transformed.base_values = base_value    
    shap_values_transformed.data = original_shap_values.data    
    #Now returning the transformed array    
    return shap_values_transformed


plt.style.use('default')

st.set_page_config(
    page_title = 'Real-Time Fraud Detection',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>MELD评分计算器</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'> </h1>", unsafe_allow_html=True)


# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('血肌酐(mg/dl)', 0.0, 10.0, 0.0)
    a2 = st.sidebar.slider('总胆红素(mg/dl)', 0.0, 50.0, 0.0)
    a3 = st.sidebar.slider('国际标准化比值', 0.0, 6.0, 0.0)

    
    output = [a1,a2,a3]
    return output

outputdf = user_input_features()

#st.header('👉 Make predictions in real time')
colnames = ['血肌酐(mg/dl)','总胆红素(mg/dl)','国际标准化比值']
outputdf = pd.DataFrame([outputdf], columns= colnames)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)


p1 = 9.57*np.log(outputdf.iat[0,0])+3.78*np.log(outputdf.iat[0,1])+11.2*np.log(outputdf.iat[0,2])+6.43
#modify output dataframe

placeholder6 = st.empty()
with placeholder6.container():
    st.subheader('Part1: User input parameters below ⬇️')
    st.write(outputdf)


placeholder7 = st.empty()
with placeholder7.container():
    st.subheader('Part2: Output results ⬇️')
    st.write(f'MELD评分 = {p1}')

placeholder8 = st.empty()
with placeholder8.container():   
    st.subheader('Part3: Formulation ⬇️')
    st.write('MELD评分 = 9.57*Ln血肌酐(mg/dl) + 3.78*Ln总胆红素(mg/dl) + 11.2*Ln国际标准化比值 + 6.43')

