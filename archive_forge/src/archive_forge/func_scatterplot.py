from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def scatterplot(dataframe, headers, diag, size, height, width, title, **kwargs):
    """
    Refer to FigureFactory.create_scatterplotmatrix() for docstring

    Returns fig for scatterplotmatrix without index

    """
    dim = len(dataframe)
    fig = make_subplots(rows=dim, cols=dim, print_grid=False)
    trace_list = []
    for listy in dataframe:
        for listx in dataframe:
            if listx == listy and diag == 'histogram':
                trace = graph_objs.Histogram(x=listx, showlegend=False)
            elif listx == listy and diag == 'box':
                trace = graph_objs.Box(y=listx, name=None, showlegend=False)
            elif 'marker' in kwargs:
                kwargs['marker']['size'] = size
                trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', showlegend=False, **kwargs)
                trace_list.append(trace)
            else:
                trace = graph_objs.Scatter(x=listx, y=listy, mode='markers', marker=dict(size=size), showlegend=False, **kwargs)
            trace_list.append(trace)
    trace_index = 0
    indices = range(1, dim + 1)
    for y_index in indices:
        for x_index in indices:
            fig.append_trace(trace_list[trace_index], y_index, x_index)
            trace_index += 1
    for j in range(dim):
        xaxis_key = 'xaxis{}'.format(dim * dim - dim + 1 + j)
        fig['layout'][xaxis_key].update(title=headers[j])
    for j in range(dim):
        yaxis_key = 'yaxis{}'.format(1 + dim * j)
        fig['layout'][yaxis_key].update(title=headers[j])
    fig['layout'].update(height=height, width=width, title=title, showlegend=True)
    hide_tick_labels_from_box_subplots(fig)
    return fig