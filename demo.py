import streamlit as st
import pandas as pd
import requests

endpoint = 'http://localhost:12000/get_sentence_toxicity'

st.write("# Sentence Toxicity Classification Demo")

random_sentence = "I don't like Joe, he is too tall"
user_sentence = st.text_input("Please provide your sentence:", value=random_sentence)

submit_button = st.button("Submit")

if submit_button:
    response = requests.post(endpoint, json={'sentence': user_sentence})
    df = pd.DataFrame(response.json(), index=['Probability']).T
    df['Score'] = df['Probability'].gt(0.5).map({True: 1, False: 0})
    st.write("## Result:")
    st.dataframe(df)
