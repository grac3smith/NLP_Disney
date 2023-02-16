import pandas as pd
df = pd.read_excel('final_reviews.xlsx')
df.head()
df['review_text'] = df['review_text'].astype(str).str.lower()
df.head(3)
df.info()
from nltk.tokenize import RegexpTokenizer

regexp = RegexpTokenizer('\w+')

df['text_token']=df['review_text'].apply(regexp.tokenize)
df.head(3)
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Make a list of english stopwords
stopwords = nltk.corpus.stopwords.words("english")

# Extend the list with your own custom stopwords
my_stopwords = ['disney', 'park', 'time', 'day', 'get', 'parks', 'rides', 'ride', 'one', 'magic', 'world',
                'would', 'people', 'kingdom', 'experience', 'even']
stopwords.extend(my_stopwords)
# Remove stopwords
df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
df.head(3)
df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
df[['review_text', 'text_token', 'text_string']].head()
all_words = ' '.join([word for word in df['text_string']])
tokenized_words = nltk.tokenize.word_tokenize(all_words)
from nltk.probability import FreqDist

fdist = FreqDist(tokenized_words)
fdist
df['text_string_fdist'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 4 ]))
df[['review_text', 'text_token', 'text_string', 'text_string_fdist']].head()
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

wordnet_lem = WordNetLemmatizer()

df['text_string_lem'] = df['text_string_fdist'].apply(wordnet_lem.lemmatize)
# check if the columns are equal
df['is_equal']= (df['text_string_fdist']==df['text_string_lem'])
# show level count
df.is_equal.value_counts()
all_words_lem = ' '.join([word for word in df['text_string_lem']])
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

words = nltk.word_tokenize(all_words_lem)
fd = FreqDist(words)
fd.tabulate()
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df['polarity'] = df['text_string_lem'].apply(lambda x: analyzer.polarity_scores(x))
df.tail(3)
# Change data structure
df = pd.concat(
    [df.drop(['review_city', 'review_state', 'review_title', 'type_of_visit','polarity'], axis=1), 
     df['polarity'].apply(pd.Series)], axis=1)
df.head(3)
# Create new variable with sentiment "neutral," "positive" and "negative"
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x > 0.8 else 'negative')
df.head(10)

string = df.text_string_lem.str.cat(sep=' ')
from collections import Counter
tokens = nltk.word_tokenize(string)
tokens2 = [w for w in tokens if not w.lower() in stopwords] 
word_count = Counter(tokens2)
word_count

df2 = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
df2 = df2.rename(columns={'index':'Word', 0:'Count'})
df2
df2.to_csv('word_count.csv')

from nrclex import NRCLex
def emotion_freq(df):
    res1 = {'anger': 0.0, 'fear': 0.0, 'negative': 0.0, 'positive': 0.0, 'sadness': 0.0, 'trust': 0.0, 'anticipation': 0.0, 'joy': 0.0, 'disgust': 0.0, 'surprise': 0.0}
    score = NRCLex(df)
    freq = df.affect_frequencies
    for k, fq in freq.items():
      res1[k] = res1.get(k, 0.0) + fq
    return res1
def word_count(row):
    row = nltk.word_tokenize(row)
    cnt = len(row)
    return cnt
df
df.to_excel('sentiment_actual.xlsx')

str_rev = ','.join(df['text_string_lem'])
text_object = NRCLex(str_rev)
data = text_object.raw_emotion_scores
data

emotion_df = pd.DataFrame.from_dict(data, orient='index')
emotion_df = emotion_df.reset_index()
emotion_df = emotion_df.rename(columns={'index' : 'Emotion Classification' , 0: 'Emotion Count'})
emotion_df = emotion_df.sort_values(by=['Emotion Count'], ascending=False)
import plotly.express as px
fig = px.bar(emotion_df, x='Emotion Count', y='Emotion Classification', color = 'Emotion Classification', orientation='h', width = 800, height = 400)
fig.show()
text_object.affect_dict

affect_df = pd.DataFrame.from_dict(text_object.affect_dict, orient='index')
affect_df
affect_df.to_csv('affect.csv')
affect_df.head()

affect_df.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )
affect_df
affect = pd.read_csv('affect.csv')

text_object.affect_frequencies
emotion_df
emotion_df.to_excel('emotion.xlsx')
