import sys
import pandas as pd
import numpy as np
import nltk
import string

from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    df = df[:500] # to test code on smaller sample
    
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