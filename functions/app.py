from flask import Flask, request, jsonify, render_template
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from concurrent.futures import ProcessPoolExecutor
import pickle
from flask_cors import CORS

nltk.download('punkt')
nltk.download('stopwords')

def create_app():

    app = Flask(__name__)
    CORS(app)

    # URL of the pickle file in Google Cloud Storage
    url_lda = 'lda.pickle'

    # Load the contents of the URL into a bytes object
    with open(url_lda, 'rb') as f:
        lda_model = pickle.load(f)

    # URL of the pickle file in Google Cloud Storage
    url_tfidf = 'tfidf_data.pickle'

    # Load the contents of the URL into a bytes object
    with open(url_tfidf, 'rb') as f:
        vectorizer, term_matrix, feature_names = pickle.load(f)

    class TextPreprocessor():
        def remove_special_characters(self, text):
            text = re.sub(r'http\S+', '', text)
            text = re.sub('[^a-zA-Z\s]', '', text.lower())
            return text

        def tokenize(self, text):
            return nltk.word_tokenize(text)

        def remove_punctuation(self, tokens):
            return [token for token in tokens if token not in string.punctuation]

        def remove_stopwords(self, tokens):
            stop_words = set(stopwords.words('english')).union(STOP_WORDS)
            return [token for token in tokens if token not in stop_words]

        def stem(self, tokens):
            stemmer = PorterStemmer()
            return [stemmer.stem(token) for token in tokens]

        def parallel_apply(self, data, func, n_workers=None):
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                result = list(executor.map(func, data))
            return result

    class Topic_Label():
        def __init__(self):
            self.topic_labels = {
                0: 'borderline personality disorder, bipolar',
                1: 'depression',
                2: 'interpersonal relationships, social anxiety',
                3: 'executive dysfunction',
                4: 'adhd',
                5: 'grief, loss, family issues',
                6: 'anxiety and panic attacks',
                7: 'self-harm',
                8: 'suicidal thoughts',
                9: 'education, work, financial stress'
                }
            pass
        
        def label(self, topic_distribution):
            max_index = np.argmax(np.array(topic_distribution))
            return self.topic_labels[max_index]
        
        def dict(self,topic_distribution):
            topic_probabilities = {}
            for idx, prob in enumerate(topic_distribution[0]):
                label = self.topic_labels[idx]
                topic_probabilities[label] = prob
            return topic_probabilities


    @app.route('/predict', methods=['POST'])
    def predict():
        text = request.json['text']

        # Preprocess the input text using the TextPreprocessor class
        clean_text = TextPreprocessor().remove_special_characters(text)
        tokens = TextPreprocessor().tokenize(clean_text)
        tokens_no_punct = TextPreprocessor().remove_punctuation(tokens)
        tokens_no_stop = TextPreprocessor().remove_stopwords(tokens_no_punct)
        stemmed_tokens = TextPreprocessor().stem(tokens_no_stop)
        preprocessed_text = " ".join(stemmed_tokens)

        # Vectorize the preprocessed text using the loaded vectorizer
        new_term_matrix = vectorizer.transform([preprocessed_text])

        # Predict the topic label using the LDA model
        topic_distribution = lda_model.transform(new_term_matrix)
        topic_label = Topic_Label().label(topic_distribution)
        topic_probabilities = Topic_Label().dict(topic_distribution)

        return jsonify({'topic_label': topic_label, 'probabilities': topic_probabilities})

    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
            text = request.form['text']
            clean_text = TextPreprocessor().remove_special_characters(text)
            tokens = TextPreprocessor().tokenize(clean_text)
            tokens_no_punct = TextPreprocessor().remove_punctuation(tokens)
            tokens_no_stop = TextPreprocessor().remove_stopwords(tokens_no_punct)
            stemmed_tokens = TextPreprocessor().stem(tokens_no_stop)
            preprocessed_text = " ".join(stemmed_tokens)

            # Vectorize the preprocessed text using the loaded vectorizer
            new_term_matrix = vectorizer.transform([preprocessed_text])

            # Predict the topic label using the LDA model
            topic_distribution = lda_model.transform(new_term_matrix)
            topic_label = Topic_Label().label(topic_distribution)
            topic_probabilities = Topic_Label().dict(topic_distribution)

            return render_template('result.html', topic_label=topic_label, probabilities=topic_probabilities)
        else:
            return render_template('index.html')
        
    return app