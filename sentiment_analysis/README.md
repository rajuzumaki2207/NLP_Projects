# Text Classification using RNN 
See code: (https://github.com/rajuzumaki2207/NLP_Projects/blob/master/sentiment_analysis/TextClassification_LSTM.ipynb)

Text Classification is a process involved in Sentiment Analysis. It is classification of peoples opinion or expressions into different sentiments. Sentiments include Positive, Neutral, and Negative, Review Ratings and Happy, Sad. Sentiment Analysis can be done on different consumer centered industries to analyse people's opinion on a particular product or subject.

![alt text](https://github.com/rajuzumaki2207/NLP_Projects/blob/master/sentiment_analysis/SENTIMENT.jpg)


##1. About data
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

Content
It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

ids: The id of the tweet ( 2087)

date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

flag: The query (lyx). If there is no query, then this value is NO_QUERY.

user: the user that tweeted (robotickilldozr)

text: the text of the tweet (Lyx is cool)

Acknowledgements
The official link regarding the dataset with resources about how it was generated is here
The official paper detailing the approach is here

Citation: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

Inspiration
To detect severity from tweets. You may have a look at this.

https://www.kaggle.com/kazanova/sentiment140


## 2. Texts preprocessing and data cleaning

Tweet texts often consists of other user mentions, hyperlink texts, emoticons and punctuations. In order to use them for learning using a Language Model. I removed such words writing a function using regex. 
```python
text_clean = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text, stem=False):
  text = re.sub(text_clean, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stopwords:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)
```
## 3. Train and Split
Used simple SKLEARN to divide the dataset into test and train data set
```python
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df1, test_size =1- Train_size, random_state =7)
```

## 4 . Tokenization
**tokenizer** create tokens for every word in the data corpus and map them to a index using dictionary.

**word_index** contains the index for each word

**vocab_size** represents the total number of word in the data corpus
```python
from keras.preprocessing.text import Tokenizer

tokenizer =Tokenizer()
tokenizer.fit_on_texts(train_data.text)
word_index= tokenizer.word_index
```
Since we created tokenizer object, which can be convert any word into a key of dictionary mapped to a number.

We are going to build a sequence model. We should feed in a sequence of numbers to it. And our text column has variation of count of words. We will ensure no variance by using padding
```python
from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text), maxlen = Max_seq_len)

```


## 5. Word Embedding

Word Embedding is one of the popular representation of document vocabulary.It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc. Basically, it's a feature vector representation of words which are used for other natural language processing applications.
I could train the embedding ourselves but that would take a while to train. So I m going in the path of Computer Vision, here use Transfer Learning

The pretrained Word Embedding like GloVe & Word2Vec gives more insights for a word which can be used for classification.
```python
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM)) # initilize embedding matrix

for word ,i in word_index.items():
  embedding_vector = embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]= embedding_vector

```

## 6. Using LSTM for Training
As one can see in the word cloud, the some words are predominantly feature in both Positive and Negative tweets. This could be a problem if I use a Machine Learning model like Naive Bayes, SVD, etc. Hence I am going to use a sequence model.

RNN can handle a sequence of data and learn a pattern of input sequence or scalar value as output.

For Model architecture

1. Embedding Layer: For generating vector for each input sequence
2. ConV1D layer: Extracting features and convolve into smaller feature vector

3. LSTM: Long Short Term Memory is a variant of RNN, which has memory state cell to learn the context of words which are further along the text to carry contextual meaning.

4. Dense: Fully connected layer for classification
```python
seq_input= Input(shape=(Max_seq_len,), dtype="int32")
embedding_seq = embedding_layer(seq_input)

X = SpatialDropout1D(0.2)(embedding_seq)
X = Conv1D(64,5, activation="relu")(X)
X = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(X)

X= Dense(512, activation = "relu")(X)
X= Dropout(0.5)(X)

X= Dense(512, activation = "relu")(X)
outputs = Dense(1, activation= "sigmoid")(X)
model = Model(seq_input, outputs)
```
## 7. Model Evalulate
Model is trained and val_accuracy is comming out to be around 80%.
**It's a pretty good model we trained here in terms of NLP. Around 80% accuracy is good enough considering the baseline human accuracy also pretty low in these tasks.**

![alt text](https://github.com/rajuzumaki2207/NLP_Projects/blob/master/sentiment_analysis/loss.png)