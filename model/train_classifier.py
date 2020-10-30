import sys
import pandas as pd
import numpy as np
import nltk
import string
import joblib
from sqlalchemy import create_engine
from collections import defaultdict
import pprint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download([
    'stopwords', 
    'averaged_perceptron_tagger', 
    'maxent_ne_chunker',
    'words'
])

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation
tk = nltk.word_tokenize
pos_tag_sent = nltk.pos_tag_sents
lm = nltk.WordNetLemmatizer()


def load_data(database_filepath):
    '''
    load data from sql data base
    database_filepath - path of database
    table_name - "MyTable" (see process_data.save_data())
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("MyTable", engine)  
#     df = df[:500] # to test code on smaller sample
    
    X = df['message']
    y = df.drop(columns=['message', 'original', 'genre'])
    
    return X, y, y.columns.tolist()


def tokenize(text):
    '''
    break text into list of tokens and lemmatize it
    clean from punctuation and stopwords
    clean_lem - list of cleaned and lemmatized words
    '''
    
    tokens = [
        word.lower() 
        for word in tk(text) 
        if (word not in punctuation) and (word not in stopwords)
    ]
    clean_lem = [lm.lemmatize(word) for word in tokens]
    
    return clean_lem


class FeatureCount(BaseEstimator, TransformerMixin):
    
    def text_2_sentences(self, X):
        '''
        split X (pd.Series containing text) by sentences, tokenize and clean words
        better performance on nltk.pos when organized in sentences batch
        return df - DataFrame text, list of words and list of clean words.
        '''
        
        if isinstance(X, str):
            X = [X]
            
        df = pd.DataFrame(X, columns=['message'])
        df['words'] = df['message'].apply(tk)
        df['clean_words'] = df['words'].apply(
            lambda s: [
                word.lower().strip()
                for word in s
                if (word not in stopwords) and (word not in punctuation)
            ]
        )
        
        return df
    
    
    def count_pos(self, df):
        '''
        count number of noun per row of sent_df
        count number of cardinal per row sent_df
        return DataFrame with NN count, and CD count per message
        '''
        df['POS'] = pos_tag_sent(df['words'].tolist())
        df['NN_count'] = df['POS'].apply(lambda s: np.sum([word[1].startswith('NN') for word in s]))
        df['CD_count'] = df['POS'].apply(lambda s: np.sum([word[1]=='CD' for word in s]))

        return df[['NN_count', 'CD_count']]
    
    
    def count_misc(self, df):
        '''
        Perform miscellaneous counts on text
        char_count - number of characters per message
        stop_count - number of common stop words per message
        punc_count - number of punctuation per message
        word_mean_len - average words length per message
        return DataFrame with all features
        '''
        df['char_count'] = df['message'].apply(lambda s: len(s))
        df['stop_count'] = df['words'].apply(lambda s: len([w for w in s if w in stopwords]))
        df['punc_count'] = df['words'].apply(lambda s: len([w for w in s if w in punctuation]))
        df['word_mean_len'] = df['clean_words'].apply(lambda s: np.mean([len(w) for w in s]))

        
        return df[['char_count', 'stop_count', 'punc_count', 'word_mean_len']]
    
    
    def fit(self, x, y=None):
        return self
    
    
    def transform(self, X):
        '''
        concatenate count features into a DataFrame
        '''
        df = self.text_2_sentences(X)
        pos_count = self.count_pos(df)
        other_count = self.count_misc(df)
        
        output = pd.concat([pos_count, other_count], axis=1)
        
        return output
         
    

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