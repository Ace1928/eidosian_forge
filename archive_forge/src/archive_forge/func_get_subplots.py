import json
import warnings
import os
from plotly import exceptions, optional_imports
from plotly.files import PLOTLY_DIR
def get_subplots(rows=1, columns=1, print_grid=False, **kwargs):
    """Return a dictionary instance with the subplots set in 'layout'.

    Example 1:
    # stack two subplots vertically
    fig = tools.get_subplots(rows=2)
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x1', yaxis='y1')]
    fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

    Example 2:
    # print out string showing the subplot grid you've put in the layout
    fig = tools.get_subplots(rows=3, columns=2, print_grid=True)

    Keywords arguments with constant defaults:

    rows (kwarg, int greater than 0, default=1):
        Number of rows, evenly spaced vertically on the figure.

    columns (kwarg, int greater than 0, default=1):
        Number of columns, evenly spaced horizontally on the figure.

    horizontal_spacing (kwarg, float in [0,1], default=0.1):
        Space between subplot columns. Applied to all columns.

    vertical_spacing (kwarg, float in [0,1], default=0.05):
        Space between subplot rows. Applied to all rows.

    print_grid (kwarg, True | False, default=False):
        If True, prints a tab-delimited string representation
        of your plot grid.

    Keyword arguments with variable defaults:

    horizontal_spacing (kwarg, float in [0,1], default=0.2 / columns):
        Space between subplot columns.

    vertical_spacing (kwarg, float in [0,1], default=0.3 / rows):
        Space between subplot rows.

    """
    from plotly.graph_objs import graph_objs
    warnings.warn('tools.get_subplots is depreciated. Please use tools.make_subplots instead.')
    if not isinstance(rows, int) or rows <= 0:
        raise Exception("Keyword argument 'rows' must be an int greater than 0")
    if not isinstance(columns, int) or columns <= 0:
        raise Exception("Keyword argument 'columns' must be an int greater than 0")
    VALID_KWARGS = ['horizontal_spacing', 'vertical_spacing']
    for key in kwargs.keys():
        if key not in VALID_KWARGS:
            raise Exception("Invalid keyword argument: '{0}'".format(key))
    try:
        horizontal_spacing = float(kwargs['horizontal_spacing'])
    except KeyError:
        horizontal_spacing = 0.2 / columns
    try:
        vertical_spacing = float(kwargs['vertical_spacing'])
    except KeyError:
        vertical_spacing = 0.3 / rows
    fig = dict(layout=graph_objs.Layout())
    plot_width = (1 - horizontal_spacing * (columns - 1)) / columns
    plot_height = (1 - vertical_spacing * (rows - 1)) / rows
    plot_num = 0
    for rrr in range(rows):
        for ccc in range(columns):
            xaxis_name = 'xaxis{0}'.format(plot_num + 1)
            x_anchor = 'y{0}'.format(plot_num + 1)
            x_start = (plot_width + horizontal_spacing) * ccc
            x_end = x_start + plot_width
            yaxis_name = 'yaxis{0}'.format(plot_num + 1)
            y_anchor = 'x{0}'.format(plot_num + 1)
            y_start = (plot_height + vertical_spacing) * rrr
            y_end = y_start + plot_height
            xaxis = dict(domain=[x_start, x_end], anchor=x_anchor)
            fig['layout'][xaxis_name] = xaxis
            yaxis = dict(domain=[y_start, y_end], anchor=y_anchor)
            fig['layout'][yaxis_name] = yaxis
            plot_num += 1
    if print_grid:
        print('This is the format of your plot grid!')
        grid_string = ''
        plot = 1
        for rrr in range(rows):
            grid_line = ''
            for ccc in range(columns):
                grid_line += '[{0}]\t'.format(plot)
                plot += 1
            grid_string = grid_line + '\n' + grid_string
        print(grid_string)
    return graph_objs.Figure(fig)