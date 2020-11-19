import pandas as pd
import plotly.graph_objects as go

if False:
    cat_class = pd.DataFrame()

    for col in y:
        cat_class[col] = y[col].value_counts(sort=False, normalize=True)

    cat_class = cat_class.T # transpose dataframe for stackplot

    # test plot
    if False:
        cat_class.sort_values(by=[0,1]).plot(
            kind='barh',
            stacked=True,
            colormap=ListedColormap(sns.color_palette("colorblind", 10))
        )
        # plt.show()


# plotly
def get_cat(cat):
    cat_class = pd.DataFrame()
    for col in cat:
        cat_class[col] = cat[col].value_counts(sort=False, normalize=True)

    return cat_class.T 
    pass


def bar_stack(data):
    t1 = go.Bar(
        x=list(data.index),
        y=data[0],
        name='0'
    )

    t2 = go.Bar(
        x=list(data.index),
        y=data[1],
        name='1'
    )

    t3 = go.Bar(
        x=list(data.index),
        y=data[2],
        name='2'
    )


    data=[t1, t2, t3]
    
    fig = go.Figure(
        data=data,
        layout={
            'barmode': 'stack',
            'title': {
                'text': 'Percentage of each class per Category',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'title': 'Category'
            },
            'yaxis': {
                'tickformat': ',.0%'
            }
        }
    )
    
    return fig
