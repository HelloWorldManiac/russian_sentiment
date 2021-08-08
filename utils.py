import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

class Punctuator:
    def __init__(self, encoder_path, regex, maxlen, padding):
        """
            encoder_path -- path to the pickled encoder
            regex        -- regex for cleansing
            maxlen       -- hyperparam from the trainig stage
            padding      -- "post" for the purposes of this model
        """
        self.encoder = pickle.load(open(encoder_path, "rb"))
        self.punct = re.compile(regex)
        self.padding = padding
        self.maxlen = maxlen
    
    def _depunctuate(self, x): return self.punct.sub(string=x, repl=" ").strip()
    
    def preprocess(self, text):
        """
        tokenize and encode raw text
        text -- message string
        """
        text = self._depunctuate(text.lower())
        text = [self.encoder.encode(text)]
        return pad_sequences(text, maxlen=self.maxlen, padding=self.padding)
