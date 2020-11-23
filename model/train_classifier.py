import sys
import pandas as pd
import numpy as np
import nltk
import string
import joblib

from collections import defaultdict
import pprint

from sqlalchemy import create_engine 

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from utils.utils import tokenize, FeatureCount


def load_data(database_filepath):
    '''
    load data from sql data base
    database_filepath - path of database
    table_name - "MyTable" (see process_data.save_data())
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("MyTable", engine)  
    df = df[:1000] # to test code on smaller sample
    
    X = df['message']
    y = df.drop(columns=['message', 'original', 'genre'])
    
    return X, y, y.columns.tolist()
      
    
def build_model():
    '''
    create model by steps
    1. build features TF-IDF + words counts (scaled between 0-1)
    2. create classifier
    3. set model parameters to be tested (including test of different classifier)
    4. perform grid search with cross-validation
    '''
    pipe1 = Pipeline([
        ('features', FeatureUnion([
            ('text_pipe', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('count_pipe', Pipeline([
                ('counter', FeatureCount()),
                ('scaler', MinMaxScaler())
            ]))
             
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    # specify parameters for grid search
    # set parameters for pipeline with "name__"
    params1 = [
        {
            'clf__estimator': [RandomForestClassifier(n_jobs=-1)],
            'clf__estimator__min_samples_leaf': [1],
            'clf__estimator__bootstrap': [False],
            'clf__estimator__n_estimators': [200],
            'clf__estimator__max_features': ['sqrt']
        },
        {
            'clf__estimator': [AdaBoostClassifier()],
            'clf__estimator__learning_rate':[0.5, 1],
            'clf__estimator__n_estimators':[10, 25, 50]
        }
    ]
    
    # to build model faster
    params2 = [{
        'clf__estimator': [RandomForestClassifier(n_jobs=-1)],
        'clf__estimator__min_samples_leaf': [1],
        'clf__estimator__bootstrap': [False],
        'clf__estimator__n_estimators': [200],
        'clf__estimator__max_features': ['sqrt']
    }]
    
    cv = GridSearchCV(pipe1, params2, cv=5) # remove n_job=-1 to avoid memory leak (check clf)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    get precision, recall and f1-score.
    we present the 'weighted' result as they account for label imbalance.
    
    res - DataFrame of averaged precision, recall and f1-score for each category
    '''
    
    
    y_pred = model.predict(X_test)
    df_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
    res = defaultdict(dict)
    acc = []

    for col in category_names:
        report = classification_report(Y_test[col].values, df_pred[col].values, output_dict=True)
        report['weighted avg'].pop('support', None)
        res[col] = report['weighted avg']
                
    res = pd.DataFrame.from_dict(res, orient='index')
    print(res)
    print(f'Total Average:\n{res.mean()}')
    
    return res


def save_model(model, model_filepath):
    
    with open(model_filepath+'.z','wb') as f: 
        joblib.dump(model.best_estimator_, f, compress=True)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n\tDATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('\tBest Parameters:\n')
        pprint.pprint(model.best_params_)
        print(f'\tBest Score:{model.best_score_:.3}', '\n')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n\tMODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    