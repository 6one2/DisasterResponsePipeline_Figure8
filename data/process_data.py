import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def clean_category(df):
    '''
    separate category name from record from category DataFrame
    cat - new DataFrame one column per category
    '''
    # extracts categories columns from first row of categories
    cat_col = [x.split('-')[0] for x in df.iloc[0,-1].split(';')]
    
    # creates new categories with 1 column per category
    cat = df['categories'].str.split(';', expand=True)
    cat.columns = cat_col
    
    # remove category name from each observation
    for col in cat:
        cat[col] = cat[col].str.get(-1).astype(int)
        
    return cat


def load_data(messages_filepath, categories_filepath):
    '''
    load and merge messages and category classification
    df - merged DataFrame
    '''
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    
    df = messages.merge(clean_category(categories), 
                        on='id',
                        how='outer'
                       )
    
    return df


def clean_data(df):
    '''
    remove empty rows
    remove duplicates
    (considering removing categrories with low information (only 1 label for example))
    n_df - cleaned DataFrame
    '''
    #remove empty messages:
    mask = df['message'].apply(lambda t: re.findall('\w',t)!=[]).values
    df = df[mask]
    
    #remove extra-classes (to keep binary data)
    cat = df.drop(columns=['message', 'original', 'genre'])
    mask = (cat.values>=2).any(1) # find rows with extra-class
    df = df[~mask] # remove rows with extra-class
        
    
    #remove duplicates
    print(f'Number of duplicates before: {df.duplicated(keep="first").sum()}')
    n_df = df.drop_duplicates()
    print(f'Number of duplicates after: {n_df.duplicated(keep="first").sum()}')
    
    return n_df


def save_data(df, database_filename):
    '''
    create a table in database_filename with a pandas DataFrame.
    
    df - pandas DataFrame
    path - local path of database
    table_name - "MyTable"
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    if not engine.dialect.has_table(engine, "MyTable"):
        df.to_sql("MyTable", engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()