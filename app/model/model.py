import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
__version__ = "0.1.0"
BASE_DIR = Path(__file__).resolve(strict=True).parent
with open(f"{BASE_DIR}/tse_rf_model.pkl", "rb") as f:
      model = pickle.load(f)
# converts audio clip into feature points. 'path' is the file path to the audio clip
def get_feature_points(path):
  feature_dict= dict()
  y, sr = librosa.load(path)
  pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)
  max_indexes = np.argmax(magnitudes, axis=0)
  # get the pitches of the max indexes per time slice
  pitches = pitches[max_indexes, range(magnitudes.shape[1])]
  feature_dict['name']=[path]
  feature_dict['pitch_max'] = [pitches.max()]
  feature_dict['pitch_min'] = [pitches.min()]
  feature_dict['pitch_mean'] = [pitches.mean()]
  feature_dict['pitch_std'] = [pitches.std()]
  feature_dict['pitch_var'] = [pitches.var()]
  feature_dict['pitch_range'] = [feature_dict['pitch_max'][0]-feature_dict['pitch_min'][0]]
  feature_dict['pitch_changerate'] = [feature_dict['pitch_range'][0]/len(pitches)]
  # print(pitch_max,pitch_min, pitch_mean, pitch_std, pitch_var, pitch_range, pitch_changerate)


  #converting into power levels(dB)
  Xdb = librosa.amplitude_to_db(abs(y))
  feature_dict['power_max'] = [Xdb.max()]
  feature_dict['power_min'] = [Xdb.min()]
  feature_dict['power_mean'] = [Xdb.mean()]
  feature_dict['power_std'] = [Xdb.std()]
  feature_dict['power_var'] = [Xdb.var()]
  feature_dict['power_range'] = [feature_dict['power_max'][0]-feature_dict['pitch_min'][0]]
  feature_dict['power_changerate'] = [feature_dict['power_range'][0]/len(Xdb)]

  stft = np.abs(librosa.stft(y))

  for idx,i in enumerate(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)):
      feature_dict['mfccs'+str(idx)] = [i]
  for idx,i in enumerate(np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)):
      feature_dict['chroma'+str(idx)] = [i]
  for idx,i in enumerate(np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)):
      feature_dict['mel'+str(idx)] = [i]
  for idx,i in enumerate(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)):
      feature_dict['contrast'+str(idx)] = [i]
  for idx,i in enumerate(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T,axis=0)):
      feature_dict['tonnetz'+str(idx)] = [i]

  frame = pd.DataFrame.from_dict(feature_dict)
  return frame.drop(['name'], axis=1)

EMOTION_CODE = {0: 'fear', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'anger', 5:'disgust', 6:'neutral',7:'boredom', 8:'calm'}



def predict_pipeline(audio_file):
   
   
    features = get_feature_points(audio_file)
    pred = model.predict(features)
    return [EMOTION_CODE[x] for x in pred]


    
