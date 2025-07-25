#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !apt update & apt install -y espeak-ng
# !pip install faster-whisper pydub


# In[2]:


import os
import re
import gc
import string
import shutil
import traceback
import phonemizer
import pandas as pd
from tqdm import tqdm

from faster_whisper import WhisperModel
from pydub import AudioSegment


# In[3]:


get_ipython().system('rm single-word-voices/train/*')


# In[4]:


src = '/storage/CDM/hf-ft/notebooks/StyleTTS2-lite/Data_Speech'
model = WhisperModel("base", device="cpu", num_workers=1)


# In[5]:


# dir(model)


# In[6]:


def remove_punctuation(text):
    words = re.sub('[' + re.escape(''.join(string.punctuation)) + ']', ' ', str(text)).split()
    new = ' '.join(x for x in words)
    return new

def get_phoneme(text, lang):
    try:
        my_phonemizer = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True,  with_stress=True, language_switch='remove-flags')
        return my_phonemizer.phonemize([text])[0]
    except Exception as e:
        print(e)


# In[7]:


def delete_tmp():
    tmp_dir = "/tmp"
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error al eliminar {file_path}: {e}")


# In[ ]:


data = {
    'audio_path': [],
    'phonemes': [],
    'set': [],
    'split': []
}

for set_ in ['train', 'val']:
    with open(os.path.join(src, f'{set_}_f5_centralized.txt'), 'r') as text_file:
        for line in tqdm(text_file):
            try:
                delete_tmp()
                items = line.split('|')
                data['audio_path'].append(items[0])
                data['phonemes'].append(items[1])
                data['set'].append(set_)
                data['split'].append(False)
                if os.path.exists(items[0]):
                    try:
                        segments, _ = model.transcribe(items[0], word_timestamps=True)
                        audio = AudioSegment.from_wav(items[0])
        
                        for i, segment in enumerate(segments):
                            for word_info in segment.words:
                                word = word_info.word.strip()
                                label = remove_punctuation(word.lower())
                                if len(label) >= 3:
                                    start = word_info.start * 1000  # ms
                                    end = word_info.end * 1000      # ms
                                    word_audio = audio[start:end]
                                    filename = os.path.splitext(os.path.basename(items[0]))[0]
                                    output_path = os.path.join('/home/service/StyleTTS2-lite/single-word-voices', set_, f"{filename}_{i:04d}_{label}.wav")
                                    word_audio.export(output_path, format="wav")
                                    data['audio_path'].append(output_path)
                                    data['phonemes'].append(get_phoneme(label, 'es-419'))
                                    data['set'].append(set_)
                                    data['split'].append(True)
                    except Exception as e:
                        traceback.print_exc()
            except Exception as e:
                traceback.print_exc()

for set_ in ['train', 'val']:
    with open(os.path.join(src, f'{set_}.txt'), 'r') as text_file:
        for line in tqdm(text_file):
            try:
                delete_tmp()
                items = line.split('|')
                data['audio_path'].append(items[0])
                data['phonemes'].append(items[1])
                data['set'].append(set_)
                data['split'].append(False)
                if os.path.exists(items[0]):
                    try:
                        segments, _ = model.transcribe(items[0], word_timestamps=True)
                        audio = AudioSegment.from_wav(items[0])
        
                        for i, segment in enumerate(segments):
                            for word_info in segment.words:
                                word = word_info.word.strip()
                                label = remove_punctuation(word.lower())
                                if len(label) >= 3:
                                    start = word_info.start * 1000  # ms
                                    end = word_info.end * 1000      # ms
                                    word_audio = audio[start:end]
                                    filename = os.path.splitext(os.path.basename(items[0]))[0]
                                    output_path = os.path.join('/home/service/StyleTTS2-lite/single-word-voices', set_, f"{filename}_{i:04d}_{label}.wav")
                                    word_audio.export(output_path, format="wav")
                                    data['audio_path'].append(output_path)
                                    data['phonemes'].append(get_phoneme(label, 'es-419'))
                                    data['set'].append(set_)
                                    data['split'].append(True)
                    except Exception as e:
                        traceback.print_exc()
            except Exception as e:
                traceback.print_exc()


# In[ ]:


line #236


# In[ ]:


df.to_csv('consolidated_data.csv', index=False)


# In[ ]:


df = pd.DataFrame(data).drop_duplicates().sample(frac=1.0)
df


# In[ ]:


for set_ in ['train', 'val']:
    df_tmp = df[df['set'] == set_].copy()
    with open(f'Data_Speech/{set_}.txt', 'w') as text_file:
        for idx, row in df_tmp.iterrows():
            n_line = f"{row['audio_path']}|{row['phonemes'].strip()}"
            text_file.write(f'{n_line}\n')

