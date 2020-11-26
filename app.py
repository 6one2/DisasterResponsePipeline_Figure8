import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine
# from utils.utils import FeatureCount, tokenize
from herokutils.utils import tokenize, FeatureCount

from data.plot_data import get_cat, bar_stack

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True  # enable log from Flask


# load data
database_filename = 'data/DisasterResponse.db'
# engine = create_engine('sqlite:///.../data/DisasterResponse.db')
engine = create_engine(f'sqlite:///{database_filename}')
df = pd.read_sql_table('MyTable', engine)

# load model
with open('model/dummy_classifier.pkl.z', 'rb') as f:
    model = load(f)
# model = load('./model/best_model.pkl.z', 'r+')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # get category from df
    # generate stacked barplot representing percentage of each class per category
    cat_data = get_cat(df.drop(columns=['message', 'original', 'genre']))
    cat_fig = bar_stack(cat_data.sort_values(by=[0, 1]))

    graphs.append(cat_fig)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #     app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(debug=True)


if __name__ == '__main__':
    main()
