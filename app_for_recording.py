import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Skripta pruža mogućnost snimanja 30 zvučnih datoteka koje sprema u mapu u koju je spremljena i sama skripta
sample_rate = 16000

def record_audio(file_number):
    duration = 1 
    
    with st.spinner("Recording audio {file_number}..."):
        audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
    st.success("Recording finished!")

    file_path = f'recorded_audio_{file_number}.wav'
    write(file_path, sample_rate, audio_data.flatten())

    return audio_data

def main():
    audio_data = None
    st.title("Klasifikacija zvučnih naredbi")
    
    for i in range(1, 31):
        if st.button("Record 1s audio", key=f"record_button_{i}"):
            audio_data = record_audio(i)
        
if __name__ == "__main__":
    main()