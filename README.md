# __DISASTER RESPONSE - FIGURE8__

## __Project Summary__
In this project, texts and tweets from various natural disasters in 2010 and 2012 have been provided by Figure Eight in their original language and with translation in english. All messages are labeled with their relation to 36 categories.

After cleaning and consolidating the data into a SQLite database we are extracting features and testing different Machine Learning (ML) algorithms with the intent to better predict future messages categories and potentially organize the proper response more efficiently.


## __Choosing the model (running the scripts)__
After running the ETL script `data/process_data.py` you will be able to run the ML pipeline `model/train_classifier.py`

> To run the ETL and save the database:\
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

> To run ML pipeline that trains the classifier and saves the model:\
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

The ML pipeline runs as follow:
1. creates features: check `models/train_classifier.build_model()`
 - TF-IDF on cleaned and lemmatized words
 - create custom features (count of nouns, cardinals, characters, punctuations, stopwords and word mean length) and scales the count between 0 and 1 to match TF-IDF scores.

2. A grid search tests first a RandomForest Classifier and then a AdaBoost Classifier with different parameters.

Other classifiers have been tested (LogisticResgression or Multi...) but the RandomForest always provided the best score.
> Better score with other classifiers might be achieved by cleaning the data more aggressively: the categories with very little information (very small amount of some class like in _"ChildAlone"_) could be removed from the dataset.

## __Running the App__

## __Project Structure__
 - data: Contains the ETL script to clean and consolidate the messages and the categories `process_data.py`. It also contains the different data files `messages.csv` and `categories.csv` and the database output of ETL.

## __References__
1. datasets from Figure Eight: [https://appen.com/datasets/combined-disaster-response-data/](https://appen.com/datasets/combined-disaster-response-data/)
2. udacity DataScience Nano degree: [link](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=8826748925_c&utm_term=87779570854&utm_keyword=udacity%20data%20science_e&gclid=Cj0KCQjwreT8BRDTARIsAJLI0KJ0Iz8KGYSr_fqOKe5GVRGvrGkg92N3yegM49aIK5fw1G9JrNFWlacaAgofEALw_wcB)
