import pickle
import logging
import logging.config
import numpy as np
from datetime import datetime
import tensorflow as tf
from flask import Flask, request, Response
from flask_restful import Resource, Api
from utils import Punctuator
import json

app = Flask(__name__)
api = Api(app)

# create logger from config file
logging.config.fileConfig('./src/logging.conf')
logger = logging.getLogger('requestChecker')

# instantiate the preprocessor with proper params
p = Punctuator(encoder_path="./src/encoder.pkl", 
               regex="[:;\(\)= \?!\-\.,@]+",
               maxlen=320,
               padding="post")

# dict for human readability of the output
reverse_mapper = {0:'negative', 1:'neutral', 2:'positive'}

# upload the model
model = tf.keras.models.load_model('./src/sentiment.mdl')

@app.route('/sentiment', methods=['POST'])
def main():
    # extract the content from the request
    text = request.form.get("text")
    
    # check the sanity of the request
    if not text:
        logger.error("Malformed Request")
        raise ValueError("Text must be a valid not empty string")
        
    # preprocess and classify        
    prediction = model.predict(p.preprocess(text))
    prediction = prediction.argmax(axis=1).item(0)
    sentiment = reverse_mapper[prediction]
    
    # create the body of response and send it
    resp = {
            'input': text,
            'sentiment': sentiment
           }   
    json_string = json.dumps(resp, ensure_ascii=False).encode('utf8')
    
    return Response(json_string, content_type="application/json; charset=utf-8" )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8585)
