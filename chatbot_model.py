import tensorflow as tf
import numpy as np
import pandas as pd
import random
import string
import json
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import Model

from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering


class ChatbotModel(object):

    content = {}
    tokenizer: Tokenizer
    model: any
    responses = {}
    input_shape: any
    tags = []
    inputs = []

    question_answerer = pipeline("question-answering")

    def __init__(self):
        print('')

    def loadData(self, path = 'content.json'):
        with open(path) as content:
            self.content = json.load(content)

    def train(self):
        for intent in self.content['intents']:
            self.responses[intent['tag']]=intent['responses']
            for lines in intent['input']:
                self.inputs.append(lines)
                self.tags.append(intent['tag'])

        self.tokenizer = Tokenizer(num_words=2000)
        self.tokenizer.fit_on_texts(self.inputs)
        train = self.tokenizer.texts_to_sequences(self.inputs)
        x_train = pad_sequences(train)

        le = LabelEncoder()
        y_train = le.fit_transform(self.tags)

        self.input_shape = x_train.shape[1]

        vocabulary = len(self.tokenizer.word_index)
        output_length = le.classes_.shape[0]

        i = Input(shape=(self.input_shape,))
        x = Embedding(vocabulary+1,10)(i)
        x = LSTM(10,return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(output_length,activation="softmax")(x)
        self.model  = Model(i,x)

        self.model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

        train = self.model.fit(x_train,y_train,epochs=200)

    def chatbot(self, question):
        texts_p = []
        le = LabelEncoder()
        le.fit(self.tags)

        question = [letters.lower() for letters in question if letters not in string.punctuation]
        question = ''.join(question)
        texts_p.append(question)

        question = self.tokenizer.texts_to_sequences(texts_p)
        question = np.array(question).reshape(-1)
        question = pad_sequences([question],self.input_shape)

        output = self.model.predict(question)
        output = output.argmax()

        response_tag = le.inverse_transform([output])[0]
        answer = random.choice(self.responses[response_tag])
        return answer

    def loadText(self, question, context):

        model_checkpoint = "distilbert-base-cased-distilled-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

        inputs = tokenizer(question, context, return_tensors="tf")
        outputs = model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        sequence_ids = inputs.sequence_ids()

        mask = [i != 1 for i in sequence_ids]
        mask[0] = False
        mask = tf.constant(mask)[None]

        start_logits = tf.where(mask, -10000, start_logits)
        end_logits = tf.where(mask, -10000, end_logits)

        start_probabilities = tf.math.softmax(start_logits, axis=-1)[0].numpy()
        end_probabilities = tf.math.softmax(end_logits, axis=-1)[0].numpy()
        scores = start_probabilities[:, None] * end_probabilities[None, :]

        scores = np.triu(scores)

        max_index = scores.argmax().item()
        start_index = max_index // scores.shape[1]
        end_index = max_index % scores.shape[1]
        score = scores[start_index, end_index].item()

        inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
        offsets = inputs_with_offsets["offset_mapping"]

        start_char, _ = offsets[start_index]
        _, end_char = offsets[end_index]
        answer = context[start_char:end_char]

        if(answer == "" or score < 0.3):
            return json.dumps({ "answer": self.chatbot(question), "score": score }, ensure_ascii=False) 

        return json.dumps({ "answer": answer, "score": score }, ensure_ascii=False)
       