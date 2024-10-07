from keras.models import load_model
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd
from scipy.io.wavfile import write


sample_rate = 16000
model = load_model(r'new_model.keras.pt3.h5')

#definiranje klasa
y = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up','yes']
mlb = MultiLabelBinarizer()
mlb.fit(pd.Series(y).fillna("missing").str.split(', '))
y_mlb = mlb.transform(pd.Series(y).fillna("missing").str.split(', '))

#funkcija za klasificiranje snimljenog zvuka
def classify_sound_record(samples,sample_rate):

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    new_X = spectrogram.reshape((1, spectrogram.shape[0], spectrogram.shape[1], 1))

    prediction = model.predict(new_X)
    predicted_class = np.argmax(prediction)
    predicted_word = mlb.classes_[predicted_class]
    return predicted_word

#funkcija za klasificiranje učitanog zvuka
def classify_sound_read(file_path):

    sample_rate, samples = wavfile.read(file_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    new_X = spectrogram.reshape((1, spectrogram.shape[0], spectrogram.shape[1], 1))

    prediction = model.predict(new_X)
    predicted_class = np.argmax(prediction)
    predicted_word = mlb.classes_[predicted_class]
    return predicted_word

#funkcija za snimanje zvuka
def record_audio():
    duration = 1 
    
    with st.spinner("Recording..."):
        audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
    st.success("Recording finished!")
    return audio_data

def main():
    audio_data = None
    st.title("Klasifikator zvučnih naredbi")

    audio_file = st.file_uploader("Odaberite zvučnu datoteku", type=["wav"])

    #ako je stisnuto tipkalo snimi zvuk i klasificiraj ga
    if st.button("Snimi 1 sekundu zvuka"):
        audio_data = record_audio()
        if audio_data is not None:
            prediction = classify_sound_record(audio_data.flatten(), sample_rate)
            st.write(f"Predikcija: {prediction}")
            prediction = None

    #ako je učitan zvuk klasificiraj ga
    elif audio_file is not None:
        prediction = classify_sound_read(audio_file)
        st.write(f"Predikcija: {prediction}")
        prediction = None

if __name__ == "__main__":
    main()