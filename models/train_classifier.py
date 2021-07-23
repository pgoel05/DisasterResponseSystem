# import libraries
import pickle
import sys
import nltk

import pandas as pd
import re
import pickle

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sqlalchemy import create_engine

from nltk import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
pd.set_option('display.max_columns', 500)


def load_data(database_filepath):
    """
    This function fetches the data from sql database and process it

    :param:
    database_filepath: path of database containing the data

    :return:
    X: DataFrame containing features
    Y: DataFrame containing categories
    category_names: category labels names
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # child_alone has zeroes, so it could be dropped
    df.drop(['child_alone'], axis='columns', inplace=True)

    # related 2 has very few values. It could be merged with the majority class 1
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    X = df["message"]
    Y = df[list(df.columns[4:])]

    return X,Y,Y.columns


def tokenize(text):
    """
    This function tokenizes the text

    :param:
        text : Text message to be tokenized
    :return:
        clean_tokens : cleaned list of tokens from the text
    """

    # Regex for urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extracting all the urls
    detected_urls = re.findall(url_regex, text)

    # Replace url with a placeholder
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")

    # instantiate tokenizer
    tokens = word_tokenize(text)

    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize, normalize case, and remove leading/trailing white space
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    This function builds pipeline of transformers and estimator

    :return:
    ML Pipeline based model to process text followed by a classifier.
    """

    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])

    '''
    Running grid search on the pipeline to find the best model
    Parameters actually used were 
        clf__estimator__learning_rate: [0.01, 0.1, 1.0]
        clf__estimator__n_estimators: [10, 50, 100]
    
    The grid search takes a lot of time and processing power to run, so I am passing the values returned by
    cv.best_params_ to make the program rune faster  
    '''

    parameters = {'clf__estimator__learning_rate': [0.01, 0.1, 1.0],
                  'clf__estimator__n_estimators': [10, 50, 100]
                  }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model by  applying it to a test set and prints out the model performance

    :param:
        model : ML model
        X_test : Test features
        Y_test : Test labels
        category_names : label names

    """

    y_pred = model.predict(X_test)
    class_report = classification_report(y_true= Y_test,y_pred= y_pred,target_names= category_names)

    print(class_report)


def save_model(model, model_filepath):
    """
    This function saves the created ML model in a pickle file
    
    :param:
        model : ML model
        model_filepath : path to create the pickle file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
