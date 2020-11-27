# __DISASTER RESPONSE - FIGURE8__

## __Project Summary__
In this project, texts and tweets from various natural disasters in 2010 and 2012 have been provided by Figure Eight in their original language and with translation in English. All messages are labeled with their relation to 36 categories.

After cleaning and consolidating the data into an SQLite database we are extracting features and testing different Machine Learning (ML) algorithms with the intent to better predict future message categories and potentially organize the proper response more efficiently.

## __Local installation__
 - Clone github repository: `git clone https://github.com/6one2/DisasterResponsePipeline_Figure8.git`
 - Create virtual environement: `pipenv shell` and install required packages `pipenv install`

 > For deployment purpose, a custom package ([`herokutils`](https://pypi.org/project/herokutils/)) has been created for the classes used in both the training and the prediction process phase.

## __Choosing the model (running the scripts locally)__
After running the ETL script `data/process_data.py` you will be able to run the ML pipeline `model/train_classifier.py`

> To run the ETL and save the database:\
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

> To run the ML pipeline that trains the classifier and saves the model:\
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

The ML pipeline runs as follow:
1. creates features: check `models/train_classifier.build_model()`
 - TF-IDF on cleaned and lemmatized words
 - create custom features (count of nouns, cardinals, characters, punctuations, stopwords and word mean length) and scales the counts between 0 and 1 to match TF-IDF scores.

2. A grid search tests first a RandomForest Classifier and then a AdaBoost Classifier with different parameters.

Other classifiers have been tested (LogisticResgression or MultinomialNB...) but the RandomForest always provided the best score. We decided to run `train_classifiers.py` with only `params2` in `GridsearchCV()` for speed but change to `params1` for full grid search.

The score the best estimator is printed in the console and save in a file as `model_results.md`

> Better score with other classifiers might be achieved by cleaning the data more aggressively: the categories with very little information (very small amount of some class like in _"ChildAlone"_, see class per category graph in app) could be removed from the dataset.

## __Running the App__
The app is hosted on heroku [here](https://pacific-fortress-23259.herokuapp.com).
> Because of the constraints of the free heroku dynos, this app was trained only over 5000 random samples from `DisasterResponse.db` in order to limit the size to the model.

To run the app locally, after having generated the classifier run:
`python app.py`

> verify that the name of the classifier file is correct line 26

## __Project Structure__

```
.
├── Pipfile
├── Pipfile.lock
├── Procfile
├── README.md
├── app.py
├── data
│   ├── DisasterResponse.db
│   ├── categories.csv
│   ├── messages.csv
│   ├── plot_data.py
│   └── process_data.py
├── model
│   ├── classifier.pkl.z
│   ├── model_results.md
│   └── train_classifier.py
├── nltk.txt
└── templates
    ├── go.html
    └── master.html
```

## __References__
1. datasets from Figure Eight: [https://appen.com/datasets/combined-disaster-response-data/](https://appen.com/datasets/combined-disaster-response-data/)
2. udacity DataScience Nano degree: [link](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=8826748925_c&utm_term=87779570854&utm_keyword=udacity%20data%20science_e&gclid=Cj0KCQjwreT8BRDTARIsAJLI0KJ0Iz8KGYSr_fqOKe5GVRGvrGkg92N3yegM49aIK5fw1G9JrNFWlacaAgofEALw_wcB)
