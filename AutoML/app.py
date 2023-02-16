import streamlit as st
import pandas as pd
import os
import joblib

#Importing profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML Web App")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and Pycaret. And it is damnright magic!!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
    
if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    # file = 'C://Users/ponne/OneDrive/Desktop/titanic.csv'
    if file:
        df=pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.write(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning!")
    chosen_target = st.selectbox("Select your Target",df.columns)
    if st.button("Train model"):
        setup(data=df, target=chosen_target, silent=True)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.write(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.write(compare_df)
        joblib.dump(best_model, 'best_model.joblib')
 
if choice == "Download":
     with open("best_model.joblib", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")