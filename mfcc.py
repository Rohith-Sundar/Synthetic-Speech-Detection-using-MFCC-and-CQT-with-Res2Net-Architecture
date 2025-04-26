import pandas as pd
import librosa
import numpy as np
import os


df = pd.read_csv('files.csv')

df = df[df['Set'] == 'progress']
df = df.sort_values(by=['Name'])

column_data = df.iloc[:, 0]


fixed_timesteps = 150  


def pad_or_truncate(mfcc, fixed_timesteps):
    if mfcc.shape[1] > fixed_timesteps:
        return mfcc[:, :fixed_timesteps]
    else:
        pad_width = fixed_timesteps - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

i = 1
for audio_name in column_data:
    y, sr = librosa.load("flac\\" + audio_name + ".flac", sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    mfccs_padded = pad_or_truncate(mfccs, fixed_timesteps)
    
    mfcc_filename = 'mfccs\\' + os.path.splitext(audio_name)[0] + '.npy'
    np.save(mfcc_filename, mfccs_padded)
    
    print(f"{i}: Processed {audio_name}, MFCC shape: {mfccs_padded.shape}")
    i += 1
