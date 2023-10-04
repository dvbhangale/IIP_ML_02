from heapq import heappop
from shutil import move
import streamlit as st
import shap
import pickle
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from predict import get_prediction, encoder
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('Img/sal.jpg')



with open('Models/final.pickle', "rb") as f:
    model = pickle.load(f)
dfce = shap.TreeExplainer(model)
      

def explain_model_prediction(data,dfce):
        # Calculate Shap values
        shap_values = dfce.shap_values(data)
        p = shap.force_plot(dfce.expected_value[1], shap_values[1], data)
        return p, shap_values

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.set_page_config(page_title="💵💰 Income Inequality Prediction 💰💵n",
                   page_icon="💵💰", layout="wide")
st.image(image,use_column_width='always')



#creating option list for dropdown menu
options_education = [' High school graduate', ' 12th grade no diploma', ' Children', ' Bachelors degree(BA AB BS)', ' 7th and 8th grade', ' 11th grade',
                     ' 9th grade', ' Masters degree(MA MS MEng MEd MSW MBA)', ' 10th grade', ' Associates degree-academic program', ' 1st 2nd 3rd or 4th grade',
                     ' Some college but no degree', ' Less than 1st grade', ' Associates degree-occup /vocational', ' Prof school degree (MD DDS DVM LLB JD)', ' 5th or 6th grade',
                       ' Doctorate degree(PhD EdD)']
options_is_hispanic  = [' All other', ' Mexican-American', ' Central or South American', ' Mexican (Mexicano)', ' Puerto Rican', ' Other Spanish', ' Cuban',
                        ' Do not know', ' Chicano']
options_household_stat  =  [' Householder', ' Nonfamily householder', ' Child 18+ never marr Not in a subfamily', ' Child <18 never marr not in subfamily', ' Spouse of householder',
                            ' Child 18+ spouse of subfamily RP', ' Secondary individual', ' Child 18+ never marr RP of subfamily', ' Other Rel 18+ spouse of subfamily RP',
                            ' Grandchild <18 never marr not in subfamily', ' Other Rel <18 never marr child of subfamily RP', ' Other Rel 18+ ever marr RP of subfamily',
                            ' Other Rel 18+ ever marr not in subfamily', ' Child 18+ ever marr Not in a subfamily', ' RP of unrelated subfamily', ' Child 18+ ever marr RP of subfamily',
                            ' Other Rel 18+ never marr not in subfamily', ' Child under 18 of RP of unrel subfamily', ' Grandchild <18 never marr child of subfamily RP',
                            ' Grandchild 18+ never marr not in subfamily', ' Other Rel <18 never marr not in subfamily', ' In group quarters', ' Grandchild 18+ ever marr not in subfamily',
                            ' Other Rel 18+ never marr RP of subfamily', ' Child <18 never marr RP of subfamily', ' Grandchild 18+ never marr RP of subfamily',
                            ' Spouse of RP of unrelated subfamily', ' Grandchild 18+ ever marr RP of subfamily', ' Child <18 ever marr not in subfamily', ' Child <18 ever marr RP of subfamily',
                            ' Other Rel <18 ever marr RP of subfamily', ' Grandchild 18+ spouse of subfamily RP', ' Child <18 spouse of subfamily RP', ' Other Rel <18 ever marr not in subfamily',
                            ' Other Rel <18 never married RP of subfamily', ' Other Rel <18 spouse of subfamily RP', ' Grandchild <18 ever marr not in subfamily', ' Grandchild <18 never marr RP of subfamily']



features = ['age', 'education', 'is_hispanic', 'wage_per_hour',
       'working_week_per_year', 'industry_code', 'total_employed',
       'household_stat', 'gains', 'losses', 'stocks_status',
       'importance_of_record']    



st.markdown("<h1 style='text-align: center;'>💵💰 Income Inequality Prediction 💰💵 </h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following info:")

        col1, col2 = st.columns(2)

        with col1:
      
            inp = {}
            name = st.text_input('Your/Candidate Name', 'My Friend')
            inp["age"] = st.number_input("Age of the Candidate: ",min_value=1, max_value=100, step=int, placeholder="Type a number...")
            inp["education"] = st.selectbox("Education : ", options=options_education)
            inp["is_hispanic"] = st.selectbox("Is Hispanic? : ", options=options_is_hispanic)
            inp["wage_per_hour"] = st.number_input("Wages earned per hour : ", step = float, placeholder="Type decimal value...")
            inp["working_week_per_year"] = st.slider("Total weeks worked in a year: ", min_value=0, max_value=52, value=26, format="%d")
            inp["industry_code"] = st.slider("Industry Code : ", min_value=0, max_value=51, value=26, format="%d")
        with col2:
            inp["total_employed"] = st.number_input("Total Employed : ",min_value=0, max_value=6)
            inp["household_stat"] = st.selectbox("Household Status : ", options=options_household_stat)
            inp["gains"] = st.number_input("Gains : ", step = int, placeholder="Type a number...")
            inp["losses"] = st.number_input("Losses : ", step = int, placeholder="Type a number...")
            inp["'stocks_status"] = st.number_input("Stock's Status : ", step = int, placeholder="Type a number...")
            inp["importance_of_record"] = st.number_input("Importance of record : ", step = float, placeholder="Type decimal value...")

        
        submit = st.form_submit_button("Predict if the income is above or below the limit!!")

    

    if submit:
        encoded_ip = encoder(inp)
        name = name.replace("My ","Your ")
        df = pd.DataFrame(encoded_ip,columns = features)
        pred = get_prediction(data=df, model=model)

        st.markdown("""<style> .big-font { font-family:sans-serif; color:Grey; font-size: 50px; } </style> """, unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">{name}\'s income is {pred} !!</p>', unsafe_allow_html=True)
        #st.write(f" => {pred} is predicted. <=")

        p, shap_values = explain_model_prediction(df,dfce)
        st.subheader('Income Inequality Prediction Interpretation Plot')
        st_shap(p)


if __name__ == '__main__':
    main()