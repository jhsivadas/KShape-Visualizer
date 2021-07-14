import numpy as np
from numpy.random import randint
from numpy.random import seed
import pandas as pd
import kshape.core as ks
import plotly.express as px
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

"""
-Gathering and cleaning the data-
Returns dictionary of a specified number of data sets
"""
def getData(num_sets=4, directory='UCR2018-NEW'):
    data_dict = {}

    # Searching Through Files
    for i, filename in enumerate(os.listdir(directory)):
        dir = os.path.join(directory, filename)
        files = ['f1', 'f2']
        for file in os.scandir(dir):
            if file.is_file():
                file = str(file)
                file = file[11:-2]
                if file[-5:] == 'TRAIN':
                    files[0] = file
                else:
                    files[1] = file

        train_data = pd.read_csv('{}/{}'.format(dir, files[0])).to_numpy()
        test_data = pd.read_csv('{}/{}'.format(dir, files[1])).to_numpy()
        data = np.concatenate((train_data, test_data), axis=0)[:50,1:]
        data_dict[dir[12:]] = data

        if i == num_sets:
            return data_dict

    return data_dict

"""Checks for 95 percent equivalence."""
def compare(a, b):
    count = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            count +=1
    return (count / len(a)) < .05

"""Modified K Shape Algorithm"""
def _kshape(x, k):
    m = x.shape[0]
    idx = randint(0, k, size=m)
    centroids = np.zeros((k, x.shape[1]))
    distances = np.empty((m, k))
    lst = []
    lst.append(idx)
    for _ in range(100):
        old_idx = idx
        for j in range(k):
            centroids[j] = ks._extract_shape(idx, x, j, centroids[j])

        distances = (1 - ks._ncc_c_3dim(x, centroids).max(axis=2)).T

        idx = distances.argmin(1)
        lst.append(idx)
        if compare(old_idx, idx):
            break
    return lst

"""
Makes the dataframe for current iteration for graph to display. Uses the _kshape cluster index array to
create this dataframe.
"""
def make_df(dataset, num_clusters):
    # Make the DataFrame
    column_names = ["Cluster", "Time-Series", "Iteration"]
    df = pd.DataFrame(columns=column_names)

    # Getting data for dataframe
    seed(0)
    results = _kshape(dataset, num_clusters)

    for index in range(len(results[0])):
        for iterations in range(len(results)):
            time_series = index + 1
            cluster = results[iterations][index]
            frame = {"Cluster":cluster, "Time-Series":time_series,"Iteration":iterations}
            df = df.append(frame, ignore_index=True)

    return df

"""Data Set Collection"""
time_data = getData(10)

"""Dash App"""
app = dash.Dash(__name__, prevent_initial_callbacks=True)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(className="container",
    children=[
    dcc.Store(id='session', storage_type='session'),
        html.Div(className="left", children=[
            html.H1("K-Shape Algorithm"),
            html.P("This application visualizes the K-Shape clustering algorithm,"
                   " a method designed to efficiently cluster time-series data."),
            html.A('View K-Shape Research Paper',
                   href='http://people.cs.uchicago.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf',
                   className='link',
                   target="_blank"),
            html.Br(),
            html.A('View K-Shape Source Code',
                   href='https://github.com/johnpaparrizos/kshape',
                   className='link',
                   target="_blank"),
            html.H2("Select Dataset: "),
            dcc.Dropdown(
                id='dataset',
                className='dropdown',
                options=[{'label': k, 'value': k} for k in time_data.keys()],
                value='Worms'
            ),
            html.H2("Choose number of clusters: "),
            dcc.Slider(
                id='clusters',
                className='slider',
                min=2,
                max=10,
                step=1,
                value=5,
                marks={
                    2: '2',
                    5: '5',
                    10: '10',
                }
            ),
            html.Button(id='submit-button-state',
                        n_clicks=0, children='Submit', className='btn'),
            html.P("Note: Graph may take a few seconds to load.")
        ]),
        html.Div(className="right", children=[
            html.H1("Cluster Graph"),
            dcc.Graph(id="graph", className="graph-object", figure={}),
        ])
    ]
)
"""
Callback to update the graph being displayed
"""
@app.callback(
    Output('graph', 'figure'),
    [Input('clusters', 'value'),
     Input('dataset', 'value'),
     Input('submit-button-state', 'n_clicks')]
)
def update_graph(num_clusters, dataset, clicks):
    dataset = time_data[dataset]
    df = make_df(dataset, num_clusters)
    fig = px.scatter(df, x="Time-Series", y="Cluster", animation_frame="Iteration",
                     animation_group="Time-Series", color="Time-Series",
                     size_max=55, range_x=[-10, df['Time-Series'].max() + 10],
                     range_y=[-1, df['Cluster'].max() + 1])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)