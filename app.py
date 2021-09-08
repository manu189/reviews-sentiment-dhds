#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import re
import unidecode
import nltk
import spacy
import string
import matplotlib.pyplot as plt
import emoji
from TextCleaner import clean_text
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



def predict_model(x, vectorizer, model):
    x = clean_text(x)

    x_tok = [y.lemma_ for y in  nlp(x)]

    #print(x_tok)

    x =  ' '.join(word for word in x_tok)

    #print(x)

    X_single=vectorizer.transform([x]);

    return model.predict(X_single)


happy_comment = "Muchas gracias por su comentario, en Balanz trabajamos cada dia para darle una mejor experiencia de usuario."
angry_comment =  "Sentimos mucho que haya tenido inconvenientes, a la brevedad lo contactaremos para solucionar el problema."



nlp = spacy.load("es_core_news_md")


vect_filename = 'saved_vect.sav'
model_filename = 'saved_model.sav'

# load the model from disk
loaded_vect = pickle.load(open(vect_filename, 'rb'))
loaded_model = pickle.load(open(model_filename, 'rb'))




st.set_page_config(page_title="Sentimiento", page_icon=":smile:", layout="wide")



# st.image('https://camfintech.hiringroomcampus.com/assets/media/camfintech/company_5f695d4a4b8e561c835eae32.jpg',width = 400)
st.image("https://www.digitalhouse.com/ar/logo-DH.png", width = 500)
st.title('Predictor de sentimiento')

st.text('Predicción por lote:')


uploaded_file = st.file_uploader("Elija un archivo de comentarios")
if uploaded_file is not None:
    df_comments = pd.read_csv(uploaded_file,sep=';')
    try:
        df_comments = df_comments.drop(['Subject','Date Created', "Sender's Name", 'Unread'], axis=1)
    except:
        df_comments = df_comments.drop(['Subject','Date Created', "Sender's Name"], axis=1)

    df_comments.Body = df_comments.Body.apply(lambda x:  x.split('\n')[-3])
    df_comments.Body = df_comments.Body.replace("\r", '')
    st.write(df_comments)
    
    df_comments['score'] = None
    df_comments['output_comment'] = None
    df_comments['emoji'] = None
    df_comments['Sentiment'] = None
    #s['icons'] = s['emoji'].apply(lambda x: emoji.emojize(x, use_aliases=True))
    i = 0
    for x in df_comments.Body:
        df_comments.at[i, 'score'] = predict_model(x, loaded_vect, loaded_model)
        i += 1
        
    df_comments.score = df_comments.score.astype(int)
    i = 0
    for x in df_comments.Body:
        
        if df_comments.loc[i, 'score']:
            df_comments.at[i, 'output_comment'] = happy_comment
            df_comments.at[i, 'emoji'] = ":heart_eyes:"
            df_comments.at[i, 'Sentiment'] = emoji.emojize(":heart_eyes:", use_aliases=True)
        else:
            df_comments.at[i, 'output_comment'] = angry_comment
            df_comments.at[i, 'emoji'] = ":cry:"
            df_comments.at[i, 'Sentiment'] = emoji.emojize(":cry:", use_aliases=True)
            
        i += 1
    
    df_comments = df_comments.drop(['emoji'], axis=1)
        
    st.table(df_comments)
    
    fig = px.pie(df_comments, names='Sentiment', title= "Inner App's reviews", width=800, height=400, color_discrete_sequence=["blue", "green"])
    fig.update_traces(textinfo="percent", insidetextfont= dict(color="white"))
    st.write(fig)
    
    #WordCloud de df_comments.Body
    text=' '.join(df_comments.Body.replace(' ','_').to_list())
    stop_words_custom = ['balanz','q', 'hola', 'má' 'tu', 'mis', 'brubank', 'etoro', 'broker', 'asesor', 'android', 'ios', 'y', 'a', 'de', 'muy', 'la', 'que', 'app', 'para', 'el', 'en', 'es', 'me', 'yo', 'he', 'se', 'con', 'los', 'un', 'las', 'pero', '1000', 'por', 'una','lo', 'todo', 'aplicacion', '12', '15', '24', '48', '48hs', 'mi', 'como', 'al','del', 'le', 'les', 'si','sin', 'ya', 'ahora', 'te', 'desde','balanz', 'hace', 'cuando', 'poder', 'bastante','puedo', 'mas', 'cada','esta', 'hay']
    wordcloud = WordCloud(width=800, height=400, max_font_size=200, max_words=50, background_color="white", stopwords=stop_words_custom).generate(text)

    # Display the generated image:
    fig = plt.figure(figsize=(15,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.write(fig)
#     plt.show()

st.text('Prediga su comentario:')
default_value_goes_here = ""
user_input = st.text_input("Ingrese su comentario", default_value_goes_here)
if st.button('enviar') and (user_input is not None):
    score_num = predict_model(user_input, loaded_vect, loaded_model)
    score_num = score_num.astype(int)
    if score_num:
        st.markdown(':heart_eyes:')
        answer = happy_comment
    else:
        st.markdown(":cry:")
        answer = angry_comment
        
    st.write(answer)
    

# In[ ]:

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Grupo 4 - Data Science - Digital House'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
                font-size: 20px;
                color: green;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


