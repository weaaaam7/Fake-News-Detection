# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

---------------------------------Part1: Data Analysis----------------------------------

--------------------------@author: weaam almutawwa, 438201478--------------------------


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#======================================================================================#
#                                 Import libraries                                     #
#======================================================================================#

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import re

#----------------------------------------------------------------------------------------
# stopwords
nltk.download('stopwords')

#----------------------------------------------------------------------------------------
# Uploading dataset
fakedata = pd.read_table('clean_fake.txt')
realdata = pd.read_table('clean_real.txt')

fakedata.head()
realdata.head()

#----------------------------------------------------------------------------------------
# Dimensionality of our data
fakedata.shape
realdata.shape

#----------------------------------------------------------------------------------------
# labeling data to help in analysis
fakedata['label'] = 'fake'
realdata['label'] = 'real'

#----------------------------------------------------------------------------------------
# Concating data
data = pd.concat([realdata, fakedata])

#----------------------------------------------------------------------------------------
# How many fake and real headlines
print(data.groupby(['label']).count())
data.groupby(['label']).count().plot(kind="bar")
plt.show()

#----------------------------------------------------------------------------------------
# Word Cloud for the dataset | fake
file_content=open ("clean_fake.txt").read()

wordcloud = WordCloud(font_path = r'C:\Windows\Fonts\Verdana.ttf',
                            stopwords = stopwords.words('english'),
                            background_color = 'white',
                            width = 800,
                            height = 600,
                            min_font_size = 10).generate(file_content)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#----------------------------------------------------------------------------------------
# Word Cloud for the dataset | real
file_content=open ("clean_real.txt").read()

wordcloud = WordCloud(font_path = r'C:\Windows\Fonts\Verdana.ttf',
                            stopwords = stopwords.words('english'),
                            background_color = 'white',
                            width = 800,
                            height = 600,
                            min_font_size = 10).generate(file_content)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#----------------------------------------------------------------------------------------
# Function to analyze data by counting the most 10 frequent words
def wordsCount(filename):
    words = re.findall(r'\w+', open(filename).read().lower())
    stop_words = stopwords.words('english')
    wordsCleaned = [w for w in words if not w in stop_words]
    l = Counter(wordsCleaned).most_common(10)
    return l;

#----------------------------------------------------------------------------------------
# Most frequent words in fake news
fakeCount = wordsCount('clean_fake.txt')
print("Most frequent words in fake news are: ")
print(fakeCount)

#----------------------------------------------------------------------------------------
# plot frequent words in fake
n_groups = len(fakeCount)

words_fake = [x[1] for x in fakeCount]
words_count = [x[0] for x in fakeCount]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4

rects1 = plt.bar(index, words_fake, bar_width,
                 alpha=opacity,
                 label='Ocurrences')

plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.title('Occurrences by word')
plt.xticks(index + bar_width, words_count)
plt.legend()

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------
# Most frequent words in real news
realCount = wordsCount('clean_real.txt')
print("Most frequent words in real news are: ")
print(realCount)

#----------------------------------------------------------------------------------------
# plot frequent words in real
n_groups = len(realCount)

words_real = [x[1] for x in realCount]
words_count = [x[0] for x in realCount]

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4

rects1 = plt.bar(index, words_real, bar_width,
                 alpha=opacity,
                 label='Ocurrences')

plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.title('Occurrences by word')
plt.xticks(index + bar_width, words_count)
plt.legend()

plt.tight_layout()
plt.show()