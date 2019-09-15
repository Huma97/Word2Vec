import pandas as pd
import numpy as np
import utils
from tqdm import tqdm

data = pd.read_csv('~/data/lenta-ru-news.csv')

df = data[['text', 'topic']].dropna().reset_index(drop=True)

texts_as_np = df['text'].values

tokenized_texts = [ ]

for text in tqdm(texts_as_np):
    tokenized_texts.append(utils.preprocess(text, big=False))
    
np.save('tokenized_texts.npy', tokenized_texts)
