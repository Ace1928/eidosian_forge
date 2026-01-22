from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def violin_colorscale(data, data_header, group_header, colors, use_colorscale, group_stats, rugplot, sort, height, width, title):
    """
    Refer to FigureFactory.create_violin() for docstring.

    Returns fig for violin plot with colorscale.

    """
    group_name = []
    for name in data[group_header]:
        if name not in group_name:
            group_name.append(name)
    if sort:
        group_name.sort()
    for group in group_name:
        if group not in group_stats:
            raise exceptions.PlotlyError('All values/groups in the index column must be represented as a key in group_stats.')
    gb = data.groupby([group_header])
    L = len(group_name)
    fig = make_subplots(rows=1, cols=L, shared_yaxes=True, horizontal_spacing=0.025, print_grid=False)
    lowcolor = clrs.color_parser(colors[0], clrs.unlabel_rgb)
    highcolor = clrs.color_parser(colors[1], clrs.unlabel_rgb)
    group_stats_values = []
    for key in group_stats:
        group_stats_values.append(group_stats[key])
    max_value = max(group_stats_values)
    min_value = min(group_stats_values)
    for k, gr in enumerate(group_name):
        vals = np.asarray(gb.get_group(gr)[data_header], float)
        intermed = (group_stats[gr] - min_value) / (max_value - min_value)
        intermed_color = clrs.find_intermediate_color(lowcolor, highcolor, intermed)
        plot_data, plot_xrange = violinplot(vals, fillcolor='rgb{}'.format(intermed_color), rugplot=rugplot)
        layout = graph_objs.Layout()
        for item in plot_data:
            fig.append_trace(item, 1, k + 1)
        fig['layout'].update({'xaxis{}'.format(k + 1): make_XAxis(group_name[k], plot_xrange)})
    trace_dummy = graph_objs.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=2, cmin=min_value, cmax=max_value, colorscale=[[0, colors[0]], [1, colors[1]]], showscale=True), showlegend=False)
    fig.append_trace(trace_dummy, 1, L)
    fig['layout'].update({'yaxis{}'.format(1): make_YAxis('')})
    fig['layout'].update(title=title, showlegend=False, hovermode='closest', autosize=False, height=height, width=width)
    return fig