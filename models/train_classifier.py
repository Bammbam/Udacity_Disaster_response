import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import pickle

import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

from textblob import TextBlob


rel_database_filepath = '../data/Disaster_response.db'

def load_data(rel_database_filepath):
    """Read data from database"""

    engine = create_engine(f'sqlite:///{rel_database_filepath}')
    df = pd.read_sql_query("SELECT * FROM response_message", engine)
    # 'related' columns has value = 2 which is usual ==> replace it with 1
    df['related'] = np.where(df['related']==2, 1,df['related'])  
    return df['message'], df[df.columns[4:]], df.columns[4:]


def tokenize(text):
    """Transform raw text/sentence by remove stopwords, stemming, lemmatizing and, tokenize them"""
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    orig_text = text
    text = re.sub('[^A-Za-z0-9]',' ', text)
    text = text.lower()
    tok_texts = word_tokenize(text) 
    text = [stemmer.stem(lemmatizer.lemmatize(w.strip())) for w in tok_texts if w.strip() not in stopwords.words('english')]
    return text


class TextPolarizer(BaseEstimator, TransformerMixin):
    """Custom transformer that giving Polarity values of each text datum"""

    def getPolarity(self, text):
        return TextBlob(text).sentiment.polarity

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.getPolarity)
        return pd.DataFrame(X_tagged)


def build_model():
    """Combine features transformation step into single pipeline return model"""
    pipeline = Pipeline(steps=[
                   ('features', FeatureUnion([
                       (('tfidf', TfidfVectorizer(tokenizer=tokenize))),
                       ('txt_polar', TextPolarizer())
                   ])),
                   ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    """Example code to show that I can use GridSearch to tune Parameter."""
    # parameters = {'clf__estimator__min_samples_leaf': [1,2], 
    #             'clf__estimator__min_samples_split': [2,3]}

    # model = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline




def evaluate_model(model, X_test, Y_test, category_names):
    """Printing model performance metrics"""
    print(classification_report(y_pred=model.predict(X_test), y_true=Y_test, target_names=category_names))
    pass


def save_model(model, model_filepath):
    """Save the trained model for future use"""
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    try:
        rel_database_filepath, model_filepath = '../data/Disaster_response.db', './clf_model.pkl'
        print('Loading data...\n    DATABASE: {}'.format(rel_database_filepath))
        X, Y, category_names = load_data(rel_database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        # print("checking category name: ", category_names)

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    except:
        print("Something went wrong along the pipeline. Use log above to see where it went wrong")


if __name__ == '__main__':
    main()