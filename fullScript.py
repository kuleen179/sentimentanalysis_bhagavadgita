"""from google.colab import drive
drive.mount('/content/drive')"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
import re
import unicodedata
import nltk
import csv
import string
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('all')
import demoji

from sentence_transformers import SentenceTransformer

pd.options.display.max_colwidth = 250

# !pip install -q transformers

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
SEED = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler

import transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

import warnings
warnings.filterwarnings('ignore')
import operator

from sklearn.metrics import hamming_loss, jaccard_score, label_ranking_average_precision_score, f1_score
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from matplotlib.ticker import StrMethodFormatter

print(transformers.__version__)

print(transformers.__version__)

MAX_LEN = 200 #based on length of tweets
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 4
LEARNING_RATE = 1e-05 #tried 1e-03, 1e-04, 1e-05
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a padding token by default

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.tweet = dataframe['Tweet']
        self.targets = self.dataframe.list
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, index):
        tweet = str(self.tweet[index])
        tweet = " ".join(tweet.split())

        inputs = self.tokenizer(
            tweet,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        ids = inputs['input_ids'].squeeze()
        mask = inputs['attention_mask'].squeeze()

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
class GPT2Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GPT2Model.from_pretrained('gpt2')
        self.layer2 = torch.nn.Dropout(0.3)
        self.layer3 = torch.nn.Linear(self.layer1.config.hidden_size, 11)

    def forward(self, ids, mask, return_dict=False):
        outputs = self.layer1(input_ids=ids, attention_mask=mask)
        hidden_state = outputs.last_hidden_state[:, -1, :]  # Use the last token's hidden state
        out_2 = self.layer2(hidden_state)
        out_final = self.layer3(out_2)
        return out_final

model = GPT2Classifier()
model.to(device)

gpt2 = torch.load("/content/drive/MyDrive/religious_texts/gpt2model.pth")
gpt2

torch.save(model.state_dict(), '/content/drive/MyDrive/religious_texts/weights_only_gpt2.pth')

model_new = GPT2Classifier()
model_new.to(device)

model_new.load_state_dict(torch.load('/content/drive/MyDrive/religious_texts/weights_only_gpt2.pth'))

model_new

def test():
    gpt2.eval()
    gpt2_outputs = []

    with torch.no_grad():
        for unw, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = gpt2(ids, mask)

            gpt2_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return gpt2_outputs

for chapter in range(18, 19):

  new_df = pd.DataFrame()

  verses_df = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/csv folder/' + str(chapter) + '.csv')

  new_df['Tweet'] = verses_df['verse']
  values = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * len(verses_df)
  new_df['list'] = values

  test_dataset = CustomDataset(new_df, tokenizer, MAX_LEN)

  gpt2_test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }   

  test_loader = DataLoader(test_dataset, **gpt2_test_params)

  test_outputs = test()

  test_outputs = np.array(test_outputs)

  for i in range(test_outputs.shape[0]):
      for j in range(test_outputs.shape[1]):
          if test_outputs[i][j] >= 0.5: test_outputs[i][j] = 1
          else: test_outputs[i][j] = 0

  new_df['Optimistic'] = "None"
  new_df['Thankful'] = "None"
  new_df['Empathetic'] = "None"
  new_df['Pessimistic'] = "None"
  new_df['Anxious'] = "None"
  new_df['Sad'] = "None"
  new_df['Annoyed'] = "None"
  new_df['Denial'] = "None"
  new_df['Official report'] = "None"
  new_df['Surprise'] = "None"
  new_df['Joking'] = "None"
  new_df = new_df.drop(['list'], axis = 1)

  for i in range(len(test_outputs)):
    new_df['Optimistic'].iloc[i] = test_outputs[i][0]
    new_df['Thankful'].iloc[i] = test_outputs[i][1]
    new_df['Empathetic'].iloc[i] = test_outputs[i][2]
    new_df['Pessimistic'].iloc[i] = test_outputs[i][3]
    new_df['Anxious'].iloc[i] = test_outputs[i][4]
    new_df['Sad'].iloc[i] = test_outputs[i][5]
    new_df['Annoyed'].iloc[i] = test_outputs[i][6]
    new_df['Denial'].iloc[i] = test_outputs[i][7]
    new_df['Official report'].iloc[i] = test_outputs[i][8]
    new_df['Surprise'].iloc[i] = test_outputs[i][9]
    new_df['Joking'].iloc[i] = test_outputs[i][10]

    new_df.tail(5)

    new_df.to_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ' + str(chapter))

total_df = pd.DataFrame()

for chapter in range(1,19):
  df = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ' + str(chapter))

  df = df.drop(['Tweet', 'Unnamed: 0', 'Official report'], axis=1)
  df = df.apply(pd.Series.value_counts)

  df['Optimistic'] = df['Optimistic'].fillna(0)
  df['Thankful'] = df['Thankful'].fillna(0)
  df['Empathetic'] = df['Empathetic'].fillna(0)
  df['Pessimistic'] = df['Pessimistic'].fillna(0)
  df['Anxious'] = df['Anxious'].fillna(0)
  df['Sad'] = df['Sad'].fillna(0)
  df['Annoyed'] = df['Annoyed'].fillna(0)
  df['Denial'] = df['Denial'].fillna(0)
  #df['Official report'] = df['Official report'].fillna(0)
  df['Surprise'] = df['Surprise'].fillna(0)
  df['Joking'] = df['Joking'].fillna(0)

  df = df.iloc[1]

  df = pd.DataFrame(df)

  df.columns = [ 'Count']

  total_df = total_df.append(df)

total_df = total_df.reset_index()
total_df = total_df.rename(columns={"index":"Sentiment"})

new_df = pd.DataFrame({"Sentiment":["Optimistic", "Thankful", "Empathetic", "Pessimistic", "Anxious", "Sad", "Annoyed", "Denial", "Surprise", "Joking"], 
                       "Count":[0,0,0,0,0,0,0,0,0,0,]})

for ii in range(0, 10):
  for jj in range(0, 18):
    new_df.loc[ii, 'Count'] = (new_df.loc[ii, 'Count'] + total_df.loc[jj*10+ii, 'Count']).astype(int)

fig_dims = (10, 12)
fig, ax = plt.subplots(figsize=fig_dims)


sns.barplot(x = new_df.Sentiment, y = new_df.Count, ax = ax, palette = sns.color_palette("tab10"))

plt.xticks(rotation=90)

plt.ylabel('Count', labelpad=30)
plt.xlabel(' ')
# plt.title('All Chapters', fontsize = 25,  pad=25)
ax.xaxis.label.set_size(30)
ax.yaxis.label.set_size(30)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))    # format decimals on the y-axis
plt.yticks(fontsize= 25)
plt.xticks(fontsize = 25)

def all_chapter_dataframe(path):
  total_df = pd.DataFrame()

  for chapter in range(1,19):
    df = pd.read_csv(path + str(chapter))

    df = df.drop(['Tweet', 'Unnamed: 0', 'Official report'], axis=1)
    df = df.apply(pd.Series.value_counts)

    df['Optimistic'] = df['Optimistic'].fillna(0)
    df['Thankful'] = df['Thankful'].fillna(0)
    df['Empathetic'] = df['Empathetic'].fillna(0)
    df['Pessimistic'] = df['Pessimistic'].fillna(0)
    df['Anxious'] = df['Anxious'].fillna(0)
    df['Sad'] = df['Sad'].fillna(0)
    df['Annoyed'] = df['Annoyed'].fillna(0)
    df['Denial'] = df['Denial'].fillna(0)
    #df['Official report'] = df['Official report'].fillna(0)
    df['Surprise'] = df['Surprise'].fillna(0)
    df['Joking'] = df['Joking'].fillna(0)

    df = df.iloc[1]

    df = pd.DataFrame(df)

    df.columns = [ 'Count']

    total_df = total_df.append(df)
  
  total_df = total_df.reset_index()
  total_df = total_df.rename(columns={"index":"Sentiment"})

  new_df = pd.DataFrame({"Sentiment":["Optimistic", "Thankful", "Empathetic", "Pessimistic", "Anxious", "Sad", "Annoyed", "Denial", "Surprise", "Joking"], 
                       "Count":[0,0,0,0,0,0,0,0,0,0,]})

  for ii in range(0, 10):
    for jj in range(0, 18):
      new_df.loc[ii, 'Count'] = (new_df.loc[ii, 'Count'] + total_df.loc[jj*10+ii, 'Count']).astype(int)

  return new_df

eknath_easwaran = all_chapter_dataframe('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ')
mahatma_gandhi = all_chapter_dataframe('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter ')
purohit_swami = all_chapter_dataframe('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ')

df_new = pd.concat([eknath_easwaran, mahatma_gandhi['Count'], purohit_swami['Count']], axis=1)

df_new.columns = ['Sentiment', 'Eknath Easwaran', 'Mahatma Gandhi', 'Purohit Swami']

df_new = pd.melt(df_new, id_vars="Sentiment", var_name="Author", value_name="Count")

# sns.color_palette("tab10")

# sns.factorplot(x='Sentiment', y='Count', hue='Author', data=df_new, kind='bar', height=10, palette = "tab10")

colors = ["blue", "orange", "magenta"]  
myPalette = sns.xkcd_palette(colors)

sns.set_style("darkgrid")
g = sns.factorplot(x='Sentiment', y='Count', hue='Author', data=df_new, kind='bar', height=12, palette = myPalette, legend=False)

g.despine(left=True)

plt.legend(loc='upper center',prop={"size":20})

plt.ylabel('Count', labelpad=30, fontsize=30)
plt.xlabel(' ')

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))    # format decimals on the y-axis
plt.yticks(fontsize= 25)
plt.xticks(fontsize = 25)
plt.xticks(rotation=90)

def chapter_wise_dataframe(path, chapter):
  
    df = pd.read_csv(path + str(chapter))

    df = df.drop(['Tweet', 'Unnamed: 0', 'Official report'], axis=1)
    df = df.apply(pd.Series.value_counts)

    df['Optimistic'] = df['Optimistic'].fillna(0)
    df['Thankful'] = df['Thankful'].fillna(0)
    df['Empathetic'] = df['Empathetic'].fillna(0)
    df['Pessimistic'] = df['Pessimistic'].fillna(0)
    df['Anxious'] = df['Anxious'].fillna(0)
    df['Sad'] = df['Sad'].fillna(0)
    df['Annoyed'] = df['Annoyed'].fillna(0)
    df['Denial'] = df['Denial'].fillna(0)
    #df['Official report'] = df['Official report'].fillna(0)
    df['Surprise'] = df['Surprise'].fillna(0)
    df['Joking'] = df['Joking'].fillna(0)

    df = df.iloc[1]

    df = pd.DataFrame(df)

    df.columns = [ 'Count']

    return df

for chapter in range(1,19):

  eknath_easwaran = chapter_wise_dataframe('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ', chapter)
  mahatma_gandhi = chapter_wise_dataframe('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter ', chapter)
  purohit_swami = chapter_wise_dataframe('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ', chapter)

  df_new = pd.concat([eknath_easwaran, mahatma_gandhi['Count'], purohit_swami['Count']], axis=1)

  df_new.columns = ['Eknath Easwaran', 'Mahatma Gandhi', 'Purohit Swami'] 

  df_new = df_new.reset_index()

  df_new = df_new.rename(columns={"index":"Sentiment"})

  df_new = pd.melt(df_new, id_vars="Sentiment", var_name="Author", value_name="Count")

  colors = ["blue", "orange", "green"]  
  myPalette = sns.xkcd_palette(colors)

  sns.set_style("darkgrid")
  g = sns.factorplot(x='Sentiment', y='Count', hue='Author', data=df_new, kind='bar', size=12, aspect=1.2, palette = myPalette, legend=False)

  g.despine(left=True)

  # if chapter == 1 or chapter == 2 or chapter == 3 or chapter == 16 or chapter==17:
  #   plt.legend(loc='upper left',prop={"size":25})

  # else:
  #   plt.legend(loc='upper center',prop={"size":25})

  plt.legend(loc='best',prop={"size":25})

  plt.ylabel('Count', labelpad=30, fontsize=30)
  plt.xlabel(' ')

  plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))    # format decimals on the y-axis
  plt.yticks(fontsize= 35)
  plt.xticks(fontsize = 35)
  plt.xticks(rotation=90)

df_new.head()

df_new.reset_index()

#HeatMaps

heatmap_df = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter 1')
heatmap_df = heatmap_df.drop(['Tweet', 'Unnamed: 0', 'Official report'], axis=1)

for chapter in range(2, 19):
  df = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ' + str(chapter))

  df = df.drop(['Tweet', 'Unnamed: 0', 'Official report'], axis=1)

  heatmap_df = pd.concat([heatmap_df, df], axis=0)

emote_array = np.zeros((10, 10))
e2i = {
    'Optimistic' : 0, 'Thankful' : 1, 'Empathetic' : 2, 'Pessimistic' : 3, 'Anxious' : 4, 'Sad' : 5, 'Annoyed' : 6, 'Denial' : 7,
    'Surprise' : 8, 'Joking' : 9
}

for i in range(len(heatmap_df)):
    l = heatmap_df.iloc[i].tolist()
    for j in range(10):
        if l[j] == 1:
            emote_array[j][j] += 1
        for k in range(j+1, 10):                 # to avoid double counting.
            if (l[j] == 1) and (l[k] == 1):
                emote_array[j][k] += int(1)
                emote_array[k][j] += int(1)

emotions = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial', 'Surprise', 'Joking']

emote_df = pd.DataFrame(emote_array, columns = emotions)
for col in emote_df:
    emote_df[col] = emote_df[col].astype(int)

fig = plt.figure(figsize = (16, 10))
sns.set(font_scale=2)
sns.heatmap(emote_df, annot = True, cmap = 'coolwarm', xticklabels = emote_df.columns, yticklabels = emote_df.columns, 
            fmt = 'g', annot_kws = {"size" : 16})
# plt.title('Chapter ' + str(chapter), pad = 25)

#Histplots

eknath_easwaran = all_chapter_dataframe('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ')
mahatma_gandhi = all_chapter_dataframe('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter ')
purohit_swami = all_chapter_dataframe('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ')

df_new = pd.concat([eknath_easwaran, mahatma_gandhi['Count'], purohit_swami['Count']], axis=1)

df_new.columns = ['Sentiment', 'Eknath Easwaran', 'Mahatma Gandhi', 'Purohit Swami']

df_new = pd.melt(df_new, id_vars="Sentiment", var_name="Author", value_name="Count")

sns.histplot(data= df_new, x='Sentiment', hue='Author', multiple= 'stack')

df_new

#Jaccard Similarity Score

from sklearn.metrics import jaccard_similarity_score, jaccard_score

eknath_easwaran_chapter_one = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 1')
mahatma_gandhi_chapter_one = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter 1')
purohit_swami_chapter_one = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter 1')

eknath_easwaran_chapter_1 = eknath_easwaran_chapter_one.drop(['Unnamed: 0', 'Tweet', 'Official report'], axis=1)
mahatma_gandhi_chapter_1 = mahatma_gandhi_chapter_one.drop(['Unnamed: 0', 'Tweet', 'Official report'], axis=1)
purohit_swami_chapter_1 = purohit_swami_chapter_one.drop(['Unnamed: 0', 'Tweet', 'Official report'], axis=1)

row = eknath_easwaran_chapter_1.iloc[46, :]
row

purohit_swami_chapter_1.iloc[46, :]

eknath_easwaran_chapter_1

import numpy as np
from sklearn.metrics import jaccard_score
y_true = np.array([[0, 1, 1],
                   [1, 1, 0]])
y_pred = np.array([[1, 1, 1],
                   [1, 0, 0]])

# jaccard_similarity_score(y_true, y_pred, normalize=False)

for ii in range(0, eknath_easwaran_chapter_1.shape[0]):
  print(jaccard_score(eknath_easwaran_chapter_1.iloc[ii, :], purohit_swami_chapter_1.iloc[ii, :]))

df_es_mg = pd.DataFrame()
df_es_ps = pd.DataFrame()
df_mg_ps = pd.DataFrame()

chapter_list = []
es_mg_list = []
es_ps_list = []
mg_ps_list = []

sum1 =0
sum2=0
sum3=0

for chapter in range(1, 19):

  if chapter==1 or chapter==2 or chapter==4 or chapter==6 or chapter==13 or chapter==14 or chapter==18:
    continue

  eknath_easwaran_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ' + str(chapter))
  mahatma_gandhi_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter ' + str(chapter))
  purohit_swami_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ' + str(chapter))

  eknath_easwaran_chapter = eknath_easwaran_chapter.drop(['Unnamed: 0', 'Tweet', 'Official report'], axis=1)
  mahatma_gandhi_chapter = mahatma_gandhi_chapter.drop(['Unnamed: 0', 'Tweet', 'Official report'], axis=1)
  purohit_swami_chapter = purohit_swami_chapter.drop(['Unnamed: 0', 'Tweet', 'Official report'], axis=1)

  for ii in range(0, eknath_easwaran_chapter.shape[0]):
    es_ps_list.append(jaccard_score(eknath_easwaran_chapter.iloc[ii, :], purohit_swami_chapter.iloc[ii, :]))
    es_mg_list.append(jaccard_score(eknath_easwaran_chapter.iloc[ii, :], mahatma_gandhi_chapter.iloc[ii, :]))
    mg_ps_list.append(jaccard_score(purohit_swami_chapter.iloc[ii, :], mahatma_gandhi_chapter.iloc[ii, :]))

  df_es_mg1 = pd.DataFrame(chapter_list, columns=['Chapter'])
  df_es_mg2 = pd.DataFrame(es_mg_list, columns=['Easwaran Gandhi'])
  df_es_mg = pd.concat([df_es_mg1, df_es_mg2], axis=1)
  # df_es_mg.to_csv('/content/drive/MyDrive/religious_texts/jaccard score of predicted sentiments/easwaran_gandhi/chapter ' + str(chapter) + '.csv')

  df_es_ps1 = pd.DataFrame(chapter_list, columns=['Chapter'])
  df_es_ps2 = pd.DataFrame(es_ps_list, columns=['Purohit Easwaran'])
  df_es_ps = pd.concat([df_es_ps1, df_es_ps2], axis=1)
  # df_es_ps.to_csv('/content/drive/MyDrive/religious_texts/jaccard score of predicted sentiments/purohit_easwaran/chapter ' + str(chapter) + '.csv')

  df_mg_ps1 = pd.DataFrame(chapter_list, columns=['Chapter'])
  df_mg_ps2 = pd.DataFrame(mg_ps_list, columns=['Purohit Gandhi'])
  df_mg_ps = pd.concat([df_mg_ps1, df_mg_ps2], axis=1)
  # df_mg_ps.to_csv('/content/drive/MyDrive/religious_texts/jaccard score of predicted sentiments/gandhi_purohit/chapter ' + str(chapter) + '.csv')

  print('Chapter ', str(chapter), ' Average')

  print('Easwaran Gandhi ', df_es_mg['Easwaran Gandhi'].mean())
  print('Easwaran Purohit Swami ', df_es_ps['Purohit Easwaran'].mean())
  print('Purohit Swami and Gandhi ', df_mg_ps['Purohit Gandhi'].mean())

  print('\n')

  sum1+=df_es_mg['Easwaran Gandhi'].mean()
  sum2+=df_es_ps['Purohit Easwaran'].mean()
  sum3+=df_mg_ps['Purohit Gandhi'].mean()

sum1 = sum1/11
sum2 = sum2/11
sum3 = sum3/11

print('sum1 ', sum1)
print('sum2 ', sum2)
print('sum3 ', sum3)


df_es_mg1 = pd.DataFrame(chapter_list, columns=['Chapter'])
df_es_mg2 = pd.DataFrame(es_mg_list, columns=['Easwaran Gandhi'])

df_es_mg = pd.concat([df_es_mg1, df_es_mg2], axis=1)

df_es_mg

df_es_ps1 = pd.DataFrame(chapter_list, columns=['Chapter'])
df_es_ps2 = pd.DataFrame(es_ps_list, columns=['Purohit Easwaran'])

df_es_ps = pd.concat([df_es_ps1, df_es_ps2], axis=1)

df_es_ps

df_mg_ps1 = pd.DataFrame(chapter_list, columns=['Chapter'])
df_mg_ps2 = pd.DataFrame(mg_ps_list, columns=['Purohit Gandhi'])

df_mg_ps = pd.concat([df_mg_ps1, df_mg_ps2], axis=1)

df_mg_ps

df_final = pd.concat([df_es_mg, df_es_ps['Purohit Easwaran'], df_mg_ps['Purohit Gandhi']], axis=1)

df_final

Easwaran_Gandhi_avg_score = df_final['Easwaran Gandhi'].mean()
Purohit_Easwaran_avg_score = df_final['Purohit Easwaran'].mean()
Purohit_Gandhi_avg_score = df_final['Purohit Gandhi'].mean()

print(Easwaran_Gandhi_avg_score)
print(Purohit_Easwaran_avg_score)
print(Purohit_Gandhi_avg_score)

data=pd.melt(df_final, ['Chapter'])

data


sns.set_style("darkgrid")

fig_dims = (12, 6)
fig, ax = plt.subplots(figsize=fig_dims)
# sns.lineplot(x='Chapter', y='value', hue='variable', data=data, ax = ax)

sns.lineplot(x='Chapter',y='value',  hue = 'variable', data=data,
             palette = 'hot', dashes= False, marker= 'o', ax=ax)

sns.relplot(x="Chapter", y="value", hue="variable",
            dashes=False, markers=True, kind="line", data=data)

# Variation of Arjuna and Krishna's Sentiments throughout the text

sentiment_polarity = {
    'Surprise': 0,
    'Joking': 0,
    'Pessimistic': -1,
    'Anxious': -1,
    'Sad':  -1,
    'Annoyed': -1,
    'Denial': -1,
    'Optimistic': 1,
    'Thankful': 1,
    'Empathetic': 1
}

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Plotly to create interactive graph
import chart_studio.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import plotly.graph_objs as go


sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Plotly to create interactive graph
import chart_studio.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=False)
import plotly.figure_factory as ff
import plotly.graph_objs as go

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# To remove un-necessary warnings
import warnings
warnings.filterwarnings("ignore")

barcolors = ['#87B88C','#9ED2A1','#E7E8CB','#48A0C9','#2A58A1','#2E8B55','#DF3659','Grey']

# Array of verses in which Arjuna has spoken

array_of_array = [[20,21,22, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45], [3,4,5,6,7, 53], [0,1,35], [3], [0], [32,33,36,37,38], [], [0,1], [], 
                  [11,12,13,14,15,16,17], [0,1,2,3,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,35,36,37,38,39,40,41,42,43,44,45],[0], [],[20],[],[],[0],
                  [0,72]]

polarity = 0
count = 0

polarity_array = []
chapter = 1
df_sentiments = pd.DataFrame(columns=['Chapter', 'Polarity'])

for array in array_of_array:

  eknath_easwaran_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ' + str(chapter))

  polarity = 0
  if len(array)!=0:

    for element in array:
      df1 = eknath_easwaran_chapter.loc[element, :]
      if df1['Optimistic'] == 1 :
        count+=1
        polarity+=1

      if df1['Thankful'] == 1 :
        count+=1
        polarity+=1

      if df1['Empathetic'] == 1 :
        count+=1
        polarity+=1

      if df1['Denial'] == 1 :
        count+=1
        polarity -= 1 

      if df1['Annoyed'] == 1 :
        count+=1
        polarity -= 1

      if df1['Sad'] == 1 :
        count+=1
        polarity -= 1

      if df1['Anxious'] == 1 :
        count+=1
        polarity -= 1

      if df1['Pessimistic'] == 1 :
        count+=1
        polarity -= 1

      if df1['Joking'] == 1 :
        #count+=1
        polarity+=0

      if df1['Surprise'] == 1 :
        #count+=1
        polarity+=0

  df_temp = pd.DataFrame([[chapter, polarity]], columns = ['Chapter', 'Polarity'])
  df_sentiments = pd.concat([df_temp, df_sentiments], axis=0, ignore_index = True)
  chapter+=1

df_sentiments = df_sentiments.sort_values('Chapter')

df_sentiments.reset_index()
df_sentiments.set_index('Chapter')


plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(16,5))
p6=sns.lineplot(x = df_sentiments['Chapter'], y = df_sentiments['Polarity'], label='Arjuna\'s Sentiments')
p6.set_xlabel("Verses Where Arjuna Speaks")

plt.ylabel('Polarity', labelpad=40, fontsize=30)
plt.xlabel('Chapter', labelpad=30, fontsize=30)

# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))    # format decimals on the y-axis
plt.yticks(fontsize= 25)
plt.xticks(np.arange(1, 19, step=1), fontsize = 25)

plt.show()

def flatten(lst):
    new_list = []
    for sublist in lst:
      if isinstance(sublist, list):
        for item in sublist:
          new_list.append(item)
      else:
        new_list.append(sublist)

    return new_list     

array = flatten([2,3, range(11, 54), range(55, 73)])
print(array)

array_of_array

# Array of verses in which Lord Krishna has spoken

array_of_array = [[], flatten([2,3, list(np.arange(11, 54)), list(np.arange(55, 71))]), flatten([list(np.arange(3,36)), list(np.arange(37,43))]), 
                    flatten([1,2,3, list(np.arange(5, 43))]),
                      flatten([list(np.arange(2, 30))]),
                        flatten([list(np.arange(1,33)), 35,36, list(np.arange(40,48))]),
                          flatten([list(np.arange(1,31))]),
                            flatten([list(np.arange(2,29))]),
                              flatten([list(np.arange(1,35))]),
                                flatten([list(np.arange(1,12)), list(np.arange(19,43))]),
                                  flatten([5,6,7,8, 32,33,34,47,48,49,52,53,54,55]),
                                    flatten([list(np.arange(2,21))]),
                                      flatten([list(np.arange(1,35))]),
                                        flatten([list(np.arange(1,21)),22,23,24,25,26,27]),
                                          flatten([list(np.arange(1,21))]),
                                            flatten([list(np.arange(1,25))]),
                                              flatten([list(np.arange(2,29))]),
                                                flatten([list(np.arange(2,73))])]

polarity = 0
count = 0

polarity_array = []
chapter = 1
df_sentiments = pd.DataFrame(columns=['Chapter', 'Polarity'])

for array in array_of_array:

  eknath_easwaran_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ' + str(chapter))

  polarity = 0
  if len(array)!=0:

    for element in array:
      df1 = eknath_easwaran_chapter.loc[element-1, :]
      if df1['Optimistic'] == 1 :
        count+=1
        polarity+=1

      if df1['Thankful'] == 1 :
        count+=1
        polarity+=1

      if df1['Empathetic'] == 1 :
        count+=1
        polarity+=1

      if df1['Denial'] == 1 :
        count+=1
        polarity -= 1 

      if df1['Annoyed'] == 1 :
        count+=1
        polarity -= 1

      if df1['Sad'] == 1 :
        count+=1
        polarity -= 1

      if df1['Anxious'] == 1 :
        count+=1
        polarity -= 1

      if df1['Pessimistic'] == 1 :
        count+=1
        polarity -= 1

      if df1['Joking'] == 1 :
        #count+=1
        polarity+=0

      if df1['Surprise'] == 1 :
        #count+=1
        polarity+=0

  df_temp = pd.DataFrame([[chapter, polarity]], columns = ['Chapter', 'Polarity'])
  df_sentiments = pd.concat([df_temp, df_sentiments], axis=0, ignore_index = True)
  chapter+=1

df_sentiments = df_sentiments.sort_values('Chapter')

df_sentiments.reset_index()
df_sentiments.set_index('Chapter')

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(16,5))
p6=sns.lineplot(x = df_sentiments['Chapter'], y = df_sentiments['Polarity'], label='Krishna\'s Sentiments')
p6.set_xlabel("Verses Where Shri Krishna Speaks")

plt.ylabel('Polarity', labelpad=40, fontsize=30)
plt.xlabel('Chapter', labelpad=30, fontsize=30)

# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))    # format decimals on the y-axis
plt.yticks(fontsize= 25)
plt.xticks(np.arange(1, 19, step=1), fontsize = 25)

plt.show()

eknath_easwaran_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 18')
eknath_easwaran_chapter = eknath_easwaran_chapter.drop(['Unnamed: 0', 'Official report'], axis=1)

eknath_easwaran_chapter.tail(50)

df1 = eknath_easwaran_chapter_two.loc[53, :]
df1['Optimistic']

# Pie Chart for Word Tranformations

data = [671, 397]
labels = ['M.K. Gandhi', 'Sh. Purohit Swami']

#define Seaborn color palette to use
colors = sns.color_palette('bright')[3:5]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%', counterclock = False)
plt.show()

mahatma_gandhi = pd.read_excel('/content/drive/MyDrive/religious_texts/mahatma gandhi/Mahatma Gandhi.xlsx')

sns.color_palette("tab10")

# myPalette = sns.xkcd_palette(colors)

sns.set_style("darkgrid")
g = sns.factorplot(x='Chapter No', y='Count', hue='Author', data=mahatma_gandhi, kind='bar', height=10, legend=False)

g.despine(left=True)

plt.legend(loc='upper left',prop={"size":20})

plt.ylabel('Count', labelpad=30, fontsize=30)
plt.xlabel(' ')

# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))    # format decimals on the y-axis
plt.yticks(fontsize= 25)
plt.xticks(fontsize = 25)

# Analyzing classification of Sentiments (experimentation)

df_chapter_12 = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 12')
df_chapter_12.drop(['Unnamed: 0', 'Official report'], axis=1, inplace=True)

df_gandhi_chapter_12 = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter 12')

df_chapter_12_optimistic = df_chapter_12.loc[(df_chapter_12['Optimistic'] == 1)]
df_chapter_12_annoyed = df_chapter_12.loc[(df_chapter_12['Annoyed'] == 1)]

df_chapter_12_annoyed

df_chapter_12_optimistic_gandhi = df_gandhi_chapter_12.loc[(df_gandhi_chapter_12['Optimistic'] == 1)]
df_chapter_12_annoyed_gandhi = df_gandhi_chapter_12.loc[(df_gandhi_chapter_12['Annoyed'] == 1)]

df_chapter_12_annoyed_gandhi

df_gandhi_chapter_2 = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter 2')

df_gandhi_chapter_2.head(50)

df_easwaran_chapter_2 = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 2')

df_easwaran_chapter_2.head(50)

# Bigrams - Trigrams

import gensim
from gensim import utils
import nltk
nltk.download('stopwords')

topic_df = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 1')
topic_df = topic_df.drop(['Unnamed: 0', 'Official report'], axis=1)

for chapter in range(2, 19):
  df = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ' + str(chapter))

  df = df.drop(['Unnamed: 0', 'Official report'], axis=1)

  topic_df = pd.concat([topic_df, df], axis=0)

  topic_df2 = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter 1')
topic_df2 = topic_df2.drop(['Unnamed: 0', 'Official report'], axis=1)

for chapter in range(2, 19):
  df2 = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter ' + str(chapter))

  df2 = df2.drop(['Unnamed: 0', 'Official report'], axis=1)

  topic_df2 = pd.concat([topic_df2, df2], axis=0)

topic_df2

# Preprocessing functions

# Stopwords
stop_words = stopwords.words('english')
newStopWords = ["shri", "replied", "asked", "shall", "lord", "said"]
stop_words.extend(newStopWords)

def remove_stopwords(tweets):
    return [[word for word in gensim.utils.simple_preprocess(str(tweet)) if word not in stop_words] for tweet in tweets]

def tokenize(tweet):
    for word in tweet:
        yield(gensim.utils.simple_preprocess(str(word), deacc=True))


def preprocessing(df):

  df['Tweet'] = df['Tweet'].str.lower()                                                      # Convert to lowercase
  df['Tweet'] = df['Tweet'].str.replace("[^a-zA-Z#]", " ")                                   # Remove punctuations
  df['Tweet'] = df['Tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))     # Remove short words
  df['Tweet tokens'] = list(tokenize(df['Tweet']))
  df['tokens_no_stop'] = remove_stopwords(df['Tweet'])
  df['tokens_no_stop_joined'] = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split(' ') if word not in stop_words]))
  return df

def retrieve_text(df):
  doc = '. '.join(df['tokens_no_stop_joined'])
  return doc

def ngrams_series_func(data, n):
  
  wordList = re.sub("[^\w]", " ",  data).split()
  ngrams_series = (pd.Series(nltk.ngrams(wordList, n)).value_counts())[:10]
  return ngrams_series

# Visualization functions

from matplotlib import cm, dates
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
barcolors = ['#87B88C','#9ED2A1','#E7E8CB','#48A0C9','#2A58A1','#2E8B55','#DF3659','Grey']
barstyle = {"edgecolor":"black", "linewidth":1}
heatmap1_args = dict(annot=True, fmt='.0f', square=False, cmap=cm.get_cmap("RdGy", 10), center = 90, vmin=0, vmax=10000, lw=4, cbar=False)
heatmap2_args = dict(annot=True, fmt='.3f', square=False, cmap="Greens", center = 0.5, lw=4, cbar=False)
heatmap3_args = dict(annot=True, fmt='.0f', square=False, cmap=cmap, center = 9200, lw=4, cbar=False)

def hide_axes(this_ax):
    this_ax.set_frame_on(False)
    this_ax.set_xticks([])
    this_ax.set_yticks([])
    return this_ax

def draw_heatmap1(df,this_ax):
    hm = sns.heatmap(df, ax = this_ax, **heatmap1_args)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation=0)
    this_ax.yaxis.tick_right()
    this_ax.yaxis.set_label_position("right")
    for axis in ['top','bottom','left','right']:
        this_ax.spines[axis].set_visible(True)
        this_ax.spines[axis].set_color('black')
    return hm 

def draw_heatmap2(df,this_ax):
    hm = sns.heatmap(df, ax = this_ax, **heatmap2_args)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation=0)
    this_ax.yaxis.tick_right()
    this_ax.yaxis.set_label_position("right")
    for axis in ['top','bottom','left','right']:
        this_ax.spines[axis].set_visible(True)
        this_ax.spines[axis].set_color('black')
    return hm 

def draw_heatmap3(df,this_ax):
    hm = sns.heatmap(df, ax = this_ax, **heatmap3_args)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation=0)
    this_ax.yaxis.tick_right()
    this_ax.yaxis.set_label_position("right")
    for axis in ['top','bottom','left','right']:
        this_ax.spines[axis].set_visible(True)
        this_ax.spines[axis].set_color('black')
    return hm 

def thousands1(x, pos):
    'The two args are the value and tick position'
    return '%1.0fK' % (x * 1e-3)

formatterK1 = FuncFormatter(thousands1)

def thousands2(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x * 1e-3)

formatterK2 = FuncFormatter(thousands2)

stop_words

topic_df = preprocessing(topic_df)
topic_df

topic_df2 = preprocessing(topic_df2)
topic_df2

text = retrieve_text(topic_df)
print(text)

text2 = retrieve_text(topic_df2)
print(text2)

# uncomment the commented code for generating bigram-trigrams for single text

bigram1 = ngrams_series_func(text, 2)
trigram1 = ngrams_series_func(text, 3)

bigram2 = ngrams_series_func(text2, 2)
trigram2 = ngrams_series_func(text2, 3)

ngram1 = pd.concat([bigram1, trigram1])
ngram2 = pd.concat([bigram2, trigram2])

plt.rcParams.update({'font.size': 14})
fig, ax=plt.subplots(1,2, figsize=(16,8), gridspec_kw = {'width_ratios':[1,1], 'wspace':0.1, 'hspace':0.1})
# fig, ax=plt.subplots(1,1, figsize=(8,8))

barh_ax = ax[0]
# barh_ax = ax

ngram1[::-1].plot.barh(ax=barh_ax, color=barcolors[3],**barstyle)
barh_ax.yaxis.set_label_position("left")
barh_ax.xaxis.tick_top()
barh_ax.xaxis.set_label_position("bottom")
barh_ax.set_xlim(barh_ax.get_xlim()[::-1])
barh_ax.set_xlabel('\n Eknath Easwaran')
barh_ax.set_ylabel('',fontsize=50)

barh_ax = ax[1]
ngram2[::-1].plot.barh(ax=barh_ax, color=barcolors[6],**barstyle)
barh_ax.xaxis.tick_top()
barh_ax.xaxis.set_label_position("bottom")
barh_ax.set_xlim(barh_ax.get_xlim())
barh_ax.yaxis.tick_right()
barh_ax.set_ylabel('', fontsize=50)
barh_ax.set_xlabel('\n Mahatma Gandhi')
plt.show()

#From given Sentiments

topic_df = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 1')
topic_df = topic_df.drop(['Unnamed: 0', 'Official report'], axis=1)

for chapter in range(2, 19):
  df = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ' + str(chapter))

  df = df.drop(['Unnamed: 0', 'Official report'], axis=1)

  topic_df = pd.concat([topic_df, df], axis=0)

# Preprocessing functions

# Stopwords
stop_words = stopwords.words('english')

def remove_stopwords(tweets):
    return [[word for word in gensim.utils.simple_preprocess(str(tweet)) if word not in stop_words] for tweet in tweets]

def tokenize(tweet):
    for word in tweet:
        yield(gensim.utils.simple_preprocess(str(word), deacc=True))


def preprocessing(df):

  df['Tweet'] = df['Tweet'].str.lower()                                                      # Convert to lowercase
  df['Tweet'] = df['Tweet'].str.replace("[^a-zA-Z#]", " ")                                   # Remove punctuations
  df['Tweet'] = df['Tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))     # Remove short words
  df['Tweet tokens'] = list(tokenize(df['Tweet']))
  df['tokens_no_stop'] = remove_stopwords(df['Tweet'])
  df['tokens_no_stop_joined'] = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split(' ') if word not in stop_words]))
  return df

def retrieve_text(df):
  doc = '. '.join(df['tokens_no_stop_joined'])
  return doc

def ngrams_series_func(data, n):
  
  wordList = re.sub("[^\w]", " ",  data).split()
  ngrams_series = (pd.Series(nltk.ngrams(wordList, n)).value_counts())[:10]
  return ngrams_series

# Visualization functions

from matplotlib import cm, dates
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
barcolors = ['#87B88C','#9ED2A1','#E7E8CB','#48A0C9','#2A58A1','#2E8B55','#DF3659','Grey']
barstyle = {"edgecolor":"black", "linewidth":1}
heatmap1_args = dict(annot=True, fmt='.0f', square=False, cmap=cm.get_cmap("RdGy", 10), center = 90, vmin=0, vmax=10000, lw=4, cbar=False)
heatmap2_args = dict(annot=True, fmt='.3f', square=False, cmap="Greens", center = 0.5, lw=4, cbar=False)
heatmap3_args = dict(annot=True, fmt='.0f', square=False, cmap=cmap, center = 9200, lw=4, cbar=False)

def hide_axes(this_ax):
    this_ax.set_frame_on(False)
    this_ax.set_xticks([])
    this_ax.set_yticks([])
    return this_ax

def draw_heatmap1(df,this_ax):
    hm = sns.heatmap(df, ax = this_ax, **heatmap1_args)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation=0)
    this_ax.yaxis.tick_right()
    this_ax.yaxis.set_label_position("right")
    for axis in ['top','bottom','left','right']:
        this_ax.spines[axis].set_visible(True)
        this_ax.spines[axis].set_color('black')
    return hm 

def draw_heatmap2(df,this_ax):
    hm = sns.heatmap(df, ax = this_ax, **heatmap2_args)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation=0)
    this_ax.yaxis.tick_right()
    this_ax.yaxis.set_label_position("right")
    for axis in ['top','bottom','left','right']:
        this_ax.spines[axis].set_visible(True)
        this_ax.spines[axis].set_color('black')
    return hm 

def draw_heatmap3(df,this_ax):
    hm = sns.heatmap(df, ax = this_ax, **heatmap3_args)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation=0)
    this_ax.yaxis.tick_right()
    this_ax.yaxis.set_label_position("right")
    for axis in ['top','bottom','left','right']:
        this_ax.spines[axis].set_visible(True)
        this_ax.spines[axis].set_color('black')
    return hm 

def thousands1(x, pos):
    'The two args are the value and tick position'
    return '%1.0fK' % (x * 1e-3)

formatterK1 = FuncFormatter(thousands1)

def thousands2(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x * 1e-3)

formatterK2 = FuncFormatter(thousands2)

topic_df = preprocessing(topic_df)
topic_df

df_optimistic = topic_df[topic_df['Optimistic'] == 1]
df_pessimistic = topic_df[topic_df['Pessimistic'] == 1]
df_surprise = topic_df[topic_df['Surprise'] == 1]
df_denial = topic_df[topic_df['Denial'] == 1]
df_annoyed = topic_df[topic_df['Annoyed'] == 1]
df_thankful = topic_df[topic_df['Thankful'] == 1]
df_empathetic = topic_df[topic_df['Empathetic'] == 1]
df_anxious = topic_df[topic_df['Anxious'] == 1]
df_sad =  topic_df[topic_df['Sad'] == 1]
df_joking = topic_df[topic_df['Joking'] == 1]

text_optimistic = '. '.join(df_optimistic.tokens_no_stop_joined)
text_pessimistic = '. '.join(df_pessimistic.tokens_no_stop_joined)
text_surprise = '. '.join(df_surprise.tokens_no_stop_joined)
text_denial = '. '.join(df_denial.tokens_no_stop_joined)
text_annoyed = '. '.join(df_annoyed.tokens_no_stop_joined)
text_thankful = '. '.join(df_thankful.tokens_no_stop_joined)
text_empathetic = '. '.join(df_empathetic.tokens_no_stop_joined)
text_anxious = '. '.join(df_anxious.tokens_no_stop_joined)
text_sad = '. '.join(df_sad.tokens_no_stop_joined)
text_joking = '. '.join(df_joking.tokens_no_stop_joined)

bigram1 = ngrams_series_func(text_thankful, 2)
trigram1 = ngrams_series_func(text_thankful, 3)

# print(bigram1)

bigram2 = ngrams_series_func(text_sad, 2)
trigram2 = ngrams_series_func(text_sad, 3)

ngram1 = pd.concat([bigram1, trigram1])
ngram2 = pd.concat([bigram2, trigram2])

plt.rcParams.update({'font.size': 14})
fig, ax=plt.subplots(1,2, figsize=(16,8), gridspec_kw = {'width_ratios':[1,1], 'wspace':0.1, 'hspace':0.1})

barh_ax = ax[0]
ngram1[::-1].plot.barh(ax=barh_ax, color=barcolors[3],**barstyle)
barh_ax.yaxis.set_label_position("left")
barh_ax.xaxis.tick_top()
barh_ax.xaxis.set_label_position("bottom")
# barh_ax.xaxis.set_major_formatter(formatterK2)
# barh_ax.set_xlim([0, 1200])
barh_ax.set_xlim(barh_ax.get_xlim()[::-1])
barh_ax.set_xlabel('\n Thankful sentiment')
barh_ax.set_ylabel('',fontsize=50)

barh_ax = ax[1]
ngram2[::-1].plot.barh(ax=barh_ax, color=barcolors[6],**barstyle)
barh_ax.xaxis.tick_top()
barh_ax.xaxis.set_label_position("bottom")
# barh_ax.xaxis.set_major_formatter(formatterK2)
# barh_ax.set_xlim([0, 1200])
barh_ax.set_xlim(barh_ax.get_xlim())
barh_ax.yaxis.tick_right()
# barh_ax.set_xlabel('Tri N-gram Count - Trump Dataset', fontsize=13)
barh_ax.set_ylabel('', fontsize=50)
barh_ax.set_xlabel('\n Sad sentiment')
plt.show()

# Comparing Chapters with stark differences in Predicted Sentiments

eknath_easwaran_6 = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter 6')
mahatma_gandhi_6 = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter 6')
purohit_swami_6 = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter 6')

eknath_easwaran_6.drop(['Unnamed: 0', 'Official report'], axis=1, inplace=True)
eknath_easwaran_6

mahatma_gandhi_6

purohit_swami_6.drop(['Unnamed: 0', 'Official report'], axis=1, inplace=True)
purohit_swami_6

df_complete = pd.DataFrame()

for chapter in range(1, 19):

  if chapter==1 or chapter==2 or chapter==4 or chapter==6 or chapter==13 or chapter==14 or chapter==18:
    continue

  eknath_easwaran_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/eknath easwaran/predicted sentiment/chapter ' + str(chapter))
  mahatma_gandhi_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/mahatma gandhi/predicted sentiment/chapter ' + str(chapter))
  purohit_swami_chapter = pd.read_csv('/content/drive/MyDrive/religious_texts/purohit swami/predicted sentiment/chapter ' + str(chapter))

  eknath_easwaran_chapter['Chapter'] = chapter
  mahatma_gandhi_chapter['Chapter'] = chapter
  purohit_swami_chapter['Chapter'] = chapter

  df = pd.merge(eknath_easwaran_chapter, mahatma_gandhi_chapter, left_index=True, right_index=True)
  df = pd.merge(df, purohit_swami_chapter, left_index=True, right_index=True)

  df_complete = pd.concat([df_complete, df])

df_joking = df_complete.loc[(df_complete['Joking_x']==1) & (df_complete['Joking_y']==1) & (df_complete['Joking']==1)]
# df_joking = df[['Tweet_x', 'Tweet_y', 'Tweet']]
df_joking