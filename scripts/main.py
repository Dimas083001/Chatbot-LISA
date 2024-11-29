# awal load Library=============================================================================================================
import re  # Import library regular expression untuk manipulasi teks
import random  # Import library random untuk pemilihan respons acak
import numpy as np  # Import library numpy untuk operasi numerik
import pandas as pd  # Import library pandas untuk manipulasi data
import json  # Import library json untuk membaca data dari file JSON
import streamlit as st  # Import library streamlit untuk membuat antarmuka web
from keras._tf_keras.keras.models import load_model  # Import fungsi load_model dari TensorFlow Keras
from keras._tf_keras.keras.preprocessing.text import Tokenizer  # Import Tokenizer untuk memproses teks
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences untuk padding urutan teks
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder untuk mengonversi label kelas
# ahkir Import Library==========================================================================================================

# awal Seting Data =======================================================================================================

# Memuat data dari JSON
with open('../data/lisa.json', 'r') as f:
    data = json.load(f)  # Membaca data dari file JSON

df = pd.DataFrame(data['intents'])  # Membuat DataFrame dari data JSON

dic = {"tag": [], "patterns": [], "responses": []}  # Membuat dictionary kosong untuk menyimpan data yang akan diubah ke DataFrame
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]  # Mengambil pola (patterns) dari DataFrame
    rspns = df[df.index == i]['responses'].values[0]  # Mengambil respons dari DataFrame
    tag = df[df.index == i]['tag'].values[0]  # Mengambil tag dari DataFrame
    for j in range(len(ptrns)):
        dic['tag'].append(tag)  # Menambahkan tag ke dalam dictionary
        dic['patterns'].append(ptrns[j])  # Menambahkan pola ke dalam dictionary
        dic['responses'].append(rspns)  # Menambahkan respons ke dalam dictionary

df = pd.DataFrame.from_dict(dic)  # Membuat DataFrame baru dari dictionary

tokenizer = Tokenizer(lower=True, split=' ')  # Membuat objek Tokenizer dengan parameter tertentu
tokenizer.fit_on_texts(df['patterns'])  # Mengonversi teks pola menjadi urutan angka

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])  # Mengonversi teks pola menjadi urutan angka
X = pad_sequences(ptrn2seq, padding='post')  # Melakukan padding terhadap urutan angka

lbl_enc = LabelEncoder()  # Membuat objek LabelEncoder
y = lbl_enc.fit_transform(df['tag'])  # Mengonversi label kelas menjadi angka

# Memuat model yang telah dilatih sebelumnya
model_path = '../models/LISAv2.keras'  # Perbarui dengan path yang benar
loaded_model = load_model(model_path)  # Memuat model yang telah dilatih

# Ahkir dari Seting Data =======================================================================================================


# Aplikasi Streamlit ===========================================================================================================

response_user = []  # List untuk menyimpan respons dari pengguna
response_bot = []  # List untuk menyimpan respons dari bot


st.title("LISA")  # Menampilkan judul aplikasi chatbot
st.write("Say Hi to LISA ChatBot")


# menampilkan hasil histori dari chat sebelumnya
if "messages" not in st.session_state:
    st.session_state.messages = []  # Membuat session state untuk menyimpan histori chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # Menampilkan histori chat sebelumnya

# Dapatkan input pengguna
prompt = st.chat_input("Type your chat here")  # Menampilkan input chat dan mendapatkan prompt dari pengguna

# Proses input pengguna dan tampilkan respons
if prompt:
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', prompt)  # Menghapus karakter selain huruf dan tanda kutip dari prompt
    txt = txt.lower()  # Mengonversi prompt menjadi huruf kecil
    txt = txt.split()  # Membagi prompt menjadi kata-kata
    txt = " ".join(txt)  # Menggabungkan kata-kata kembali menjadi teks
    text.append(txt)  # Menambahkan teks ke dalam list

    x_test = tokenizer.texts_to_sequences(text)  # Mengonversi teks input pengguna menjadi urutan angka
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])  # Melakukan padding terhadap urutan angka
    y_pred = loaded_model.predict(x_test)  # Memprediksi kelas dengan model yang telah dilatih
    y_pred = y_pred.argmax()  # Mengambil indeks kelas dengan nilai probabilitas tertinggi
    tag = lbl_enc.inverse_transform([y_pred])[0]  # Mengonversi indeks kelas kembali menjadi label kelas
    responses = df[df['tag'] == tag]['responses'].values[0]  # Mengambil respons berdasarkan label kelas

    # Gunakan respons tetap daripada random.choice(responses)
    bot_response = random.choice(responses) if responses else "I cant understand what u say."  # Memilih respons bot atau respons default jika tidak ada respons yang sesuai

    with st.chat_message("user"):
        st.write(prompt)  # Menampilkan input pengguna
    with st.chat_message("assistant"):
        st.write(bot_response)  # Menampilkan respons bot
    st.session_state.messages.append({"role": "user", "content": prompt})  # Menyimpan input pengguna ke dalam histori chat
    st.session_state.messages.append({"role": "assistant", "content": "BOT: " + bot_response})  # Menyimpan respons bot ke dalam histori chat