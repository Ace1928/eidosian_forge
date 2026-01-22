from numbers import Number
import copy
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
import plotly.graph_objects as go
def gantt_colorscale(chart, colors, title, index_col, show_colorbar, bar_width, showgrid_x, showgrid_y, height, width, tasks=None, task_names=None, data=None, group_tasks=False, show_hover_fill=True):
    """
    Refer to FigureFactory.create_gantt() for docstring
    """
    if tasks is None:
        tasks = []
    if task_names is None:
        task_names = []
    if data is None:
        data = []
    showlegend = False
    for index in range(len(chart)):
        task = dict(x0=chart[index]['Start'], x1=chart[index]['Finish'], name=chart[index]['Task'])
        if 'Description' in chart[index]:
            task['description'] = chart[index]['Description']
        tasks.append(task)
    scatter_data_dict = dict()
    marker_data_dict = dict()
    if show_hover_fill:
        hoverinfo = 'name'
    else:
        hoverinfo = 'skip'
    scatter_data_template = {'x': [], 'y': [], 'mode': 'none', 'fill': 'toself', 'showlegend': False, 'hoverinfo': hoverinfo, 'legendgroup': ''}
    marker_data_template = {'x': [], 'y': [], 'mode': 'markers', 'text': [], 'marker': dict(color='', size=1, opacity=0), 'name': '', 'showlegend': False, 'legendgroup': ''}
    index_vals = []
    for row in range(len(tasks)):
        if chart[row][index_col] not in index_vals:
            index_vals.append(chart[row][index_col])
    index_vals.sort()
    if isinstance(chart[0][index_col], Number):
        if len(colors) < 2:
            raise exceptions.PlotlyError("You must use at least 2 colors in 'colors' if you are using a colorscale. However only the first two colors given will be used for the lower and upper bounds on the colormap.")
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            if not group_tasks or tn not in task_names:
                task_names.append(tn)
        if group_tasks:
            task_names.reverse()
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            del tasks[index]['name']
            groupID = index
            if group_tasks:
                groupID = task_names.index(tn)
            tasks[index]['y0'] = groupID - bar_width
            tasks[index]['y1'] = groupID + bar_width
            colors = clrs.color_parser(colors, clrs.unlabel_rgb)
            lowcolor = colors[0]
            highcolor = colors[1]
            intermed = chart[index][index_col] / 100.0
            intermed_color = clrs.find_intermediate_color(lowcolor, highcolor, intermed)
            intermed_color = clrs.color_parser(intermed_color, clrs.label_rgb)
            tasks[index]['fillcolor'] = intermed_color
            color_id = tasks[index]['fillcolor']
            if color_id not in scatter_data_dict:
                scatter_data_dict[color_id] = copy.deepcopy(scatter_data_template)
            scatter_data_dict[color_id]['fillcolor'] = color_id
            scatter_data_dict[color_id]['name'] = str(chart[index][index_col])
            scatter_data_dict[color_id]['legendgroup'] = color_id
            colors = clrs.color_parser(colors, clrs.label_rgb)
            if len(scatter_data_dict[color_id]['x']) > 0:
                scatter_data_dict[color_id]['x'].append(scatter_data_dict[color_id]['x'][-1])
                scatter_data_dict[color_id]['y'].append(None)
            xs, ys = _get_corner_points(tasks[index]['x0'], tasks[index]['y0'], tasks[index]['x1'], tasks[index]['y1'])
            scatter_data_dict[color_id]['x'] += xs
            scatter_data_dict[color_id]['y'] += ys
            if color_id not in marker_data_dict:
                marker_data_dict[color_id] = copy.deepcopy(marker_data_template)
                marker_data_dict[color_id]['marker']['color'] = color_id
                marker_data_dict[color_id]['legendgroup'] = color_id
            marker_data_dict[color_id]['x'].append(tasks[index]['x0'])
            marker_data_dict[color_id]['x'].append(tasks[index]['x1'])
            marker_data_dict[color_id]['y'].append(groupID)
            marker_data_dict[color_id]['y'].append(groupID)
            if 'description' in tasks[index]:
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                del tasks[index]['description']
            else:
                marker_data_dict[color_id]['text'].append(None)
                marker_data_dict[color_id]['text'].append(None)
        if show_colorbar is True:
            k = list(marker_data_dict.keys())[0]
            marker_data_dict[k]['marker'].update(dict(colorscale=[[0, colors[0]], [1, colors[1]]], showscale=True, cmax=100, cmin=0))
    if isinstance(chart[0][index_col], str):
        index_vals = []
        for row in range(len(tasks)):
            if chart[row][index_col] not in index_vals:
                index_vals.append(chart[row][index_col])
        index_vals.sort()
        if len(colors) < len(index_vals):
            raise exceptions.PlotlyError("Error. The number of colors in 'colors' must be no less than the number of unique index values in your group column.")
        index_vals_dict = {}
        c_index = 0
        for key in index_vals:
            if c_index > len(colors) - 1:
                c_index = 0
            index_vals_dict[key] = colors[c_index]
            c_index += 1
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            if not group_tasks or tn not in task_names:
                task_names.append(tn)
        if group_tasks:
            task_names.reverse()
        for index in range(len(tasks)):
            tn = tasks[index]['name']
            del tasks[index]['name']
            groupID = index
            if group_tasks:
                groupID = task_names.index(tn)
            tasks[index]['y0'] = groupID - bar_width
            tasks[index]['y1'] = groupID + bar_width
            tasks[index]['fillcolor'] = index_vals_dict[chart[index][index_col]]
            color_id = tasks[index]['fillcolor']
            if color_id not in scatter_data_dict:
                scatter_data_dict[color_id] = copy.deepcopy(scatter_data_template)
            scatter_data_dict[color_id]['fillcolor'] = color_id
            scatter_data_dict[color_id]['legendgroup'] = color_id
            scatter_data_dict[color_id]['name'] = str(chart[index][index_col])
            colors = clrs.color_parser(colors, clrs.label_rgb)
            if len(scatter_data_dict[color_id]['x']) > 0:
                scatter_data_dict[color_id]['x'].append(scatter_data_dict[color_id]['x'][-1])
                scatter_data_dict[color_id]['y'].append(None)
            xs, ys = _get_corner_points(tasks[index]['x0'], tasks[index]['y0'], tasks[index]['x1'], tasks[index]['y1'])
            scatter_data_dict[color_id]['x'] += xs
            scatter_data_dict[color_id]['y'] += ys
            if color_id not in marker_data_dict:
                marker_data_dict[color_id] = copy.deepcopy(marker_data_template)
                marker_data_dict[color_id]['marker']['color'] = color_id
                marker_data_dict[color_id]['legendgroup'] = color_id
            marker_data_dict[color_id]['x'].append(tasks[index]['x0'])
            marker_data_dict[color_id]['x'].append(tasks[index]['x1'])
            marker_data_dict[color_id]['y'].append(groupID)
            marker_data_dict[color_id]['y'].append(groupID)
            if 'description' in tasks[index]:
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                marker_data_dict[color_id]['text'].append(tasks[index]['description'])
                del tasks[index]['description']
            else:
                marker_data_dict[color_id]['text'].append(None)
                marker_data_dict[color_id]['text'].append(None)
        if show_colorbar is True:
            showlegend = True
            for k in scatter_data_dict:
                scatter_data_dict[k]['showlegend'] = showlegend
    layout = dict(title=title, showlegend=showlegend, height=height, width=width, shapes=[], hovermode='closest', yaxis=dict(showgrid=showgrid_y, ticktext=task_names, tickvals=list(range(len(task_names))), range=[-1, len(task_names) + 1], autorange=False, zeroline=False), xaxis=dict(showgrid=showgrid_x, zeroline=False, rangeselector=dict(buttons=list([dict(count=7, label='1w', step='day', stepmode='backward'), dict(count=1, label='1m', step='month', stepmode='backward'), dict(count=6, label='6m', step='month', stepmode='backward'), dict(count=1, label='YTD', step='year', stepmode='todate'), dict(count=1, label='1y', step='year', stepmode='backward'), dict(step='all')])), type='date'))
    data = [scatter_data_dict[k] for k in sorted(scatter_data_dict)]
    data += [marker_data_dict[k] for k in sorted(marker_data_dict)]
    fig = go.Figure(data=data, layout=layout)
    return fig