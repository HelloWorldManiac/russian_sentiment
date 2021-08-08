import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import re
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, Bidirectional, LSTM, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model

# Obtaining the data
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path")
args = parser.parse_args()
path = args.path # path to data

# the DataFrame should contain columns "label", "text" with the appropriate content
# text -- raw text of messages
# label -- sentiment labels "positive", "neutral", "negative"
# ensure correctness by dropping na and duplicates

df = pd.read_csv(path, usecols=["lable", "text"]) 
df.drop_duplicates(inplace=True) 
df.dropna(inplace = True) 

# Preprocessing: Stage 1. Data cleansing

punct = re.compile("[:;\(\)= \?!\-\.,@\n]+|\[.+\]") # compile the regex
def depunctuate(x): return punct.sub(string=x, repl=" ").strip() # oneliner to propagate over the text field
df.text = df.text.str.lower().apply(depunctuate) # purify the data

# Preprocessing: Stage 2. Tokenize and Encode

# create and build the SubwordTextEncoder
# note that the first argument is a generator
encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((t for t in df.text),
                                                                     max_subword_length=5, 
                                                                     target_vocab_size=2**12)
                                                                     
# a new field of encoded sentences
df['code'] = df.text.apply(encoder.encode) 

# Preprocessing: Stage 3. Categorical labels

# create a dict to encode labels
dct = {k:v for v,k in enumerate(base.label.unique())}
df['target'] = df.label.map(dct)

# create a dict to decode classes into human readable for further usage of the model
reverse_mapper = {v:k for k,v in dct.items()} 

#split into train/test
train_texts, test_texts, y_train, y_test = train_test_split(df.code.values, 
                                                            df.target.values, 
                                                            test_size=0.15, 
                                                            random_state=123) 


# pad the seqs
MAXLEN = max([len(t) for t in train_texts]) # maximum length of the seq, a hyperparam for Input layer

train_data = pad_sequences(train_texts, maxlen=MAXLEN, padding='post')
test_data = pad_sequences(test_texts, maxlen=MAXLEN, padding='post')

# Convert to numpy array of tf-compatible dtypes
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')


# Modelling
                                                            
# Hyperparams 
VOCAB_SIZE = encoder.vocab_size + 1
EMDEDDING_DIM = 128
N_UNITS = 64
EPOCHS = 5
BATCH_SIZE = 125

# Instantiate the layout of the model

i = Input(shape=[MAXLEN],dtype=tf.int16)

x = Embedding(VOCAB_SIZE, output_dim=EMDEDDING_DIM)(i)
x = Bidirectional(LSTM(N_UNITS, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)

o = Dense(3, activation="softmax")(x)

# Create and compile
model = Model(i,o)
model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit
model.fit(x = train_data, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15)

# Verify the quality
preds = model.predict(test_data)
y_pred = np.argmax(preds,axis=1)

f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-score is {f1}")



                                                                                                                          

