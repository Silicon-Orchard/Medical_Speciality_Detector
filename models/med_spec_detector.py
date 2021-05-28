import pickle,os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from simpletransformers.classification import ClassificationModel
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import nltk


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, 'config.json')) as config_file:
    config = json.load(config_file)

nltk.data.path.append(config['NLTK_PATH'])
nltk.download('stopwords', download_dir=config['NLTK_PATH'])
nltk.download('wordnet', download_dir=config['NLTK_PATH'])
nltk.download('punkt', download_dir=config['NLTK_PATH'])

class ModelClass:
    def clean_data(self,transcription):
        lemmatizer = WordNetLemmatizer()
        message = re.sub('[^a-zA-Z]', ' ', str(transcription))
        message = message.lower()
        message = message.split()

        message = [lemmatizer.lemmatize(word) for word in message if not word in stopwords.words('english')]
        message = ' '.join(message)
        return message

    def classical_prediction(self,transcription):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(BASE_DIR, 'config.json')) as config_file:
            config = json.load(config_file)

        with open(BASE_DIR + config["models"] + 'vectorizer.pickle', 'rb') as vectorizerhandle:
            vectorizer = pickle.load(vectorizerhandle)
            vectorizerhandle.close()

        with open(BASE_DIR + config["models"] + 'NB_model.pickle', 'rb') as NB_modelhandle:
            NB_model = pickle.load(NB_modelhandle)
            NB_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'SVC_model.pickle', 'rb') as SVC_modelhandle:
            SVC_model = pickle.load(SVC_modelhandle)
            SVC_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'KNN_model.pickle', 'rb') as KNN_modelhandle:
            KNN_model = pickle.load(KNN_modelhandle)
            KNN_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'DecisionTree_model.pickle', 'rb') as DecisionTree_modelhandle:
            DecisionTree_model = pickle.load(DecisionTree_modelhandle)
            DecisionTree_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'RandomForest_model.pickle', 'rb') as RandomForest_modelhandle:
            RandomForest_model = pickle.load(RandomForest_modelhandle)
            RandomForest_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'LogisticRegression_model.pickle', 'rb') as LogisticRegression_modelhandle:
            LogisticRegression_model = pickle.load(LogisticRegression_modelhandle)
            LogisticRegression_modelhandle.close()

        transcription = vectorizer.transform([transcription]).toarray()
        naive_bayes_prediction = NB_model.predict(transcription)
        svc_prediction = SVC_model.predict(transcription)
        knn_prediction = KNN_model.predict(transcription)
        decision_tree_prediction = DecisionTree_model.predict(transcription)
        random_forest_prediction = RandomForest_model.predict(transcription)
        logistic_regression_prediction = LogisticRegression_model.predict(transcription)

        predicted_class = [0, 0, 0, 0]
        predicted_class[naive_bayes_prediction[0]] = predicted_class[naive_bayes_prediction[0]] + 1.1
        predicted_class[svc_prediction[0]] = predicted_class[svc_prediction[0]] + 1
        predicted_class[knn_prediction[0]] = predicted_class[knn_prediction[0]] + 0.9
        predicted_class[decision_tree_prediction[0]] = predicted_class[decision_tree_prediction[0]] + 0.6
        predicted_class[random_forest_prediction[0]] = predicted_class[random_forest_prediction[0]] + 1
        predicted_class[logistic_regression_prediction[0]] = predicted_class[logistic_regression_prediction[0]] + 1.1
        return predicted_class

    def predict(self,transcription):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(BASE_DIR, 'config.json')) as config_file:
            config = json.load(config_file)
        lstm_model = load_model(BASE_DIR + config["lstm_model"])
        attention_lstm_model = load_model(BASE_DIR + config["attention_lstm_model"])
        # bert_model = ClassificationModel('bert', BASE_DIR + config['bert_model'], use_cuda=False)
        with open(BASE_DIR + config["models"] + 'tokenizer.pickle', 'rb') as tokenizer_handle:
            tokenizer = pickle.load(tokenizer_handle)
            tokenizer_handle.close()

        with open(BASE_DIR + config["models"] + 'classes.pickle', 'rb') as classes_handle:
            classes = pickle.load(classes_handle)
            classes_handle.close()

        transcription = self.clean_data(transcription)
        bert_transcription = transcription
        predicted_class = self.classical_prediction(transcription)
        transcription = tokenizer.texts_to_sequences([transcription])
        transcription = pad_sequences(transcription, padding='post', truncating='post', maxlen=120)

        # bert_prediction, raw_outputs = bert_model.predict([bert_transcription])
        attention_lstm_prediction = np.argmax(attention_lstm_model.predict(transcription), axis=-1)
        lstm_prediction = np.argmax(lstm_model.predict(transcription), axis=-1)
        # predicted_class[bert_prediction[0]] = predicted_class[bert_prediction[0]] + 1.5
        predicted_class[attention_lstm_prediction[0]] = predicted_class[attention_lstm_prediction[0]] + 0.5
        predicted_class[lstm_prediction[0]] = predicted_class[lstm_prediction[0]] + 1
        max_index = predicted_class.index(max(predicted_class))
        return classes[max_index]

    def file_prediction(self, data):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(BASE_DIR, 'config.json')) as config_file:
            config = json.load(config_file)

        with open(BASE_DIR + config["models"] + 'vectorizer.pickle', 'rb') as vectorizerhandle:
            vectorizer = pickle.load(vectorizerhandle)
            vectorizerhandle.close()

        with open(BASE_DIR + config["models"] + 'NB_model.pickle', 'rb') as NB_modelhandle:
            NB_model = pickle.load(NB_modelhandle)
            NB_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'SVC_model.pickle', 'rb') as SVC_modelhandle:
            SVC_model = pickle.load(SVC_modelhandle)
            SVC_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'KNN_model.pickle', 'rb') as KNN_modelhandle:
            KNN_model = pickle.load(KNN_modelhandle)
            KNN_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'DecisionTree_model.pickle', 'rb') as DecisionTree_modelhandle:
            DecisionTree_model = pickle.load(DecisionTree_modelhandle)
            DecisionTree_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'RandomForest_model.pickle', 'rb') as RandomForest_modelhandle:
            RandomForest_model = pickle.load(RandomForest_modelhandle)
            RandomForest_modelhandle.close()

        with open(BASE_DIR + config["models"] + 'LogisticRegression_model.pickle', 'rb') as LogisticRegression_modelhandle:
            LogisticRegression_model = pickle.load(LogisticRegression_modelhandle)
            LogisticRegression_modelhandle.close()

        lstm_model = load_model(BASE_DIR + config["keras_model"])
        with open(BASE_DIR + config["models"] + 'tokenizer.pickle', 'rb') as tokenizer_handle:
            tokenizer = pickle.load(tokenizer_handle)
            tokenizer_handle.close()

        with open(BASE_DIR + config["models"] + 'classes.pickle', 'rb') as classes_handle:
            classes = pickle.load(classes_handle)
            classes_handle.close()

        corpus = []
        for i in range(len(data)):
            corpus.append(self.clean_data(data[i]))

        lstm_transcription = tokenizer.texts_to_sequences(corpus)
        lstm_transcription = pad_sequences(lstm_transcription, padding='post', truncating='post', maxlen=120)

        transcription = vectorizer.transform(corpus).toarray()
        naive_bayes_prediction = NB_model.predict(transcription)
        svc_prediction = SVC_model.predict(transcription)
        knn_prediction = KNN_model.predict(transcription)
        decision_tree_prediction = DecisionTree_model.predict(transcription)
        random_forest_prediction = RandomForest_model.predict(transcription)
        logistic_regression_prediction = LogisticRegression_model.predict(transcription)
        bilstm_prediction = np.argmax(lstm_model.predict(lstm_transcription), axis=-1)

        ensenble_prediction = []
        for i in range(len(naive_bayes_prediction)):
            predicted_class = [0, 0, 0, 0]
            predicted_class[naive_bayes_prediction[i]] = predicted_class[naive_bayes_prediction[i]] + 1.1
            predicted_class[svc_prediction[i]] = predicted_class[svc_prediction[i]] + 1
            predicted_class[knn_prediction[i]] = predicted_class[knn_prediction[i]] + 0.9
            predicted_class[decision_tree_prediction[i]] = predicted_class[decision_tree_prediction[i]] + 0.6
            predicted_class[random_forest_prediction[i]] = predicted_class[random_forest_prediction[i]] + 1
            predicted_class[logistic_regression_prediction[i]] = predicted_class[logistic_regression_prediction[i]] + 1.1
            predicted_class[bilstm_prediction[i]] = predicted_class[bilstm_prediction[i]] + 1

            max_index = predicted_class.index(max(predicted_class))
            ensenble_prediction.append(classes[max_index])

        return ensenble_prediction