import streamlit as st
import numpy as np 
import pickle
#title for our WEB APP
st.title("Diabetes Prediction")

load_model=pickle.load(open("train_model.pkl",'rb'))
#function for pridection
def dai_pred(input_data):
    input_data = np.array(input_data).reshape(1,-1)

    prediction = load_model.predict(input_data)
    print(prediction)
    if prediction[0]==0:
        return "The Person is not Diabetic"
    else:
        return "The Person is Diabetic"

def main():
    # getting input data from user	
    Pregnancies= st.slider("Number of Prgnancies",0,18,1)
    Glucose = st.text_input("Glucose level :")
    BloodPressure=st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness:")
    Insulin = st.text_input("Insulin level :")
    BMI = st.text_input("BMI :")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Persons Age:")

    diagnosis = ""
    #button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = dai_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.header(diagnosis)
    
if __name__=='__main__':
    main()