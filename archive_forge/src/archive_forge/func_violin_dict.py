from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def violin_dict(data, data_header, group_header, colors, use_colorscale, group_stats, rugplot, sort, height, width, title):
    """
    Refer to FigureFactory.create_violin() for docstring.

    Returns fig for violin plot without colorscale.

    """
    group_name = []
    for name in data[group_header]:
        if name not in group_name:
            group_name.append(name)
    if sort:
        group_name.sort()
    for group in group_name:
        if group not in colors:
            raise exceptions.PlotlyError('If colors is a dictionary, all the group names must appear as keys in colors.')
    gb = data.groupby([group_header])
    L = len(group_name)
    fig = make_subplots(rows=1, cols=L, shared_yaxes=True, horizontal_spacing=0.025, print_grid=False)
    for k, gr in enumerate(group_name):
        vals = np.asarray(gb.get_group(gr)[data_header], float)
        plot_data, plot_xrange = violinplot(vals, fillcolor=colors[gr], rugplot=rugplot)
        layout = graph_objs.Layout()
        for item in plot_data:
            fig.append_trace(item, 1, k + 1)
        fig['layout'].update({'xaxis{}'.format(k + 1): make_XAxis(group_name[k], plot_xrange)})
    fig['layout'].update({'yaxis{}'.format(1): make_YAxis('')})
    fig['layout'].update(title=title, showlegend=False, hovermode='closest', autosize=False, height=height, width=width)
    return fig