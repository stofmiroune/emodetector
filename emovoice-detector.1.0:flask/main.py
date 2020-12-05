import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os ,glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import soundfile
import librosa.feature
import requests

app = Flask(__name__)
model = pickle.load(open('finalized_model-2.sav','rb'))



def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        #if chroma:
           # stft=np.abs(librosa.stft(X)
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            stft=np.abs(librosa.stft(X))
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result




@app.route('/messages', methods = ['POST'])
def api_message():
    f = open('./file.wav', 'wb')
    f.write(request.data)
    f.close()
    x_pred=[]
    f = open('./file.wav', 'rb')
    feature=extract_feature(f, mfcc=True, chroma=True, mel=True)
    x_pred.append(feature)
    y_pred=model.predict(x_pred)
    x = open('./x.txt', 'wb')
    x.write(y_pred)
    x.close()
    return jsonify(y_pred.tolist())
