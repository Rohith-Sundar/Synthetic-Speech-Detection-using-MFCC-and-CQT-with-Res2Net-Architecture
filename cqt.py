import pandas as pd
import librosa
import numpy as np
import os

df = pd.read_csv('files.csv')

df = df[df['Set'] == 'progress']
df = df.sort_values(by=['Name'])

column_data = df.iloc[:, 0]

fixed_timesteps = 150  

def pad_or_truncate(cqt, fixed_timesteps):
    if cqt.shape[1] > fixed_timesteps:
        return cqt[:, :fixed_timesteps]
    else:
        pad_width = fixed_timesteps - cqt.shape[1]
        return np.pad(cqt, ((0, 0), (0, pad_width)), mode='constant')

i = 1
for audio_name in column_data:
    y, sr = librosa.load("flac\\" + audio_name + ".flac", sr=None)

    CQT = librosa.cqt(y, sr=sr)

    CQT_db = librosa.amplitude_to_db(abs(CQT))

    CQT_padded = pad_or_truncate(CQT_db, fixed_timesteps)

    cqt_filename = 'cqts\\' + os.path.splitext(audio_name)[0] + '.npy'
    np.save(cqt_filename, CQT_padded)

    print(f"{i}: Processed {audio_name}, CQT shape: {CQT_padded.shape}")
    i += 1
