import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved.pkl','rb') as f:
        data=pickle.load(f)
    return data

data = load_model()
model = data["model"]
le_edu = data["le_education"]
le_gen = data["le_gender"]

def show_predict():
    st.title("Job salary prediction")
    st.write("Information to calculate job salary")

    gender = ("Male","Female")
    education = ("Bachelor's","Master's","PhD")

    age = st.slider("Age",min_value=18,max_value=100,step=1)
    gender = st.selectbox("Gender",gender)
    education = st.selectbox("Education",education)
    jobtitle = st.text_input("Job Title")
    experience = st.slider("Experience",min_value=0,max_value=70,step=1)

    ok = st.button("Calculate salary")

    if ok == True:
        X = np.array([[age,gender,education,experience]])
        X[:,1] = le_gen.transform(X[:,1])
        X[:,2] = le_edu.transform(X[:,2])
        X = X.astype(float)

        salary = model.predict(X)
        st.subheader(f"The estimated salary is Rs {salary[0]:.2f}")

show_predict()