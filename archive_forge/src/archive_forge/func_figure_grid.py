import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def figure_grid(figures_grid, row_spacing=50, column_spacing=50, share_xaxis=False, share_yaxis=False, width=None, height=None):
    """
    Construct a figure from a 2D grid of sub-figures

    Parameters
    ----------
    figures_grid: list of list of (dict or None)
        2D list of plotly figure dicts that will be combined in a grid to
        produce the resulting figure.  None values maybe used to leave empty
        grid cells
    row_spacing: float (default 50)
        Vertical spacing between rows in the grid in pixels
    column_spacing: float (default 50)
        Horizontal spacing between columns in the grid in pixels
        coordinates
    share_xaxis: bool (default False)
        Share x-axis between sub-figures in the same column. Also link all x-axes in the
        figure. This will only work if each sub-figure has a single x-axis
    share_yaxis: bool (default False)
        Share y-axis between sub-figures in the same row. Also link all y-axes in the
        figure. This will only work if each subfigure has a single y-axis
    width: int (default None)
        Final figure width. If not specified, width is the sum of the max width of
        the figures in each column
    height: int (default None)
        Final figure width. If not specified, height is the sum of the max height of
        the figures in each row

    Returns
    -------
    dict
        A plotly figure dict
    """
    row_heights = [-1 for _ in figures_grid]
    column_widths = [-1 for _ in figures_grid[0]]
    nrows = len(row_heights)
    ncols = len(column_widths)
    responsive = True
    for r in range(nrows):
        for c in range(ncols):
            fig_element = figures_grid[r][c]
            if not fig_element:
                continue
            responsive &= fig_element.get('config', {}).get('responsive', False)
    default = None if responsive else 400
    for r in range(nrows):
        for c in range(ncols):
            fig_element = figures_grid[r][c]
            if not fig_element:
                continue
            w = fig_element.get('layout', {}).get('width', default)
            if w:
                column_widths[c] = max(w, column_widths[c])
            h = fig_element.get('layout', {}).get('height', default)
            if h:
                row_heights[r] = max(h, row_heights[r])
    if width:
        available_column_width = width - (ncols - 1) * column_spacing
        column_width_scale = available_column_width / sum(column_widths)
        column_widths = [wi * column_width_scale for wi in column_widths]
    else:
        column_width_scale = 1.0
    if height:
        available_row_height = height - (nrows - 1) * row_spacing
        row_height_scale = available_row_height / sum(row_heights)
        row_heights = [hi * row_height_scale for hi in row_heights]
    else:
        row_height_scale = 1.0
    column_domains = _compute_subplot_domains(column_widths, column_spacing)
    row_domains = _compute_subplot_domains(row_heights, row_spacing)
    output_figure = {'data': [], 'layout': {}}
    for r, (fig_row, row_domain) in enumerate(zip(figures_grid, row_domains)):
        for c, (fig, column_domain) in enumerate(zip(fig_row, column_domains)):
            if fig:
                fig = copy.deepcopy(fig)
                _normalize_subplot_ids(fig)
                subplot_offsets = _get_max_subplot_ids(output_figure)
                if share_xaxis:
                    subplot_offsets['xaxis'] = c
                    if r != 0:
                        fig.get('layout', {}).pop('xaxis', None)
                if share_yaxis:
                    subplot_offsets['yaxis'] = r
                    if c != 0:
                        fig.get('layout', {}).pop('yaxis', None)
                _offset_subplot_ids(fig, subplot_offsets)
                if responsive:
                    scale_x = 1.0 / ncols
                    scale_y = 1.0 / nrows
                    px = 0.2 / ncols if ncols > 1 else 0
                    py = 0.2 / nrows if nrows > 1 else 0
                    sx = scale_x - px
                    sy = scale_y - py
                    _scale_translate(fig, sx, sy, scale_x * c + px / 2.0, scale_y * r + py / 2.0)
                else:
                    fig_height = fig['layout'].get('height', default) * row_height_scale
                    fig_width = fig['layout'].get('width', default) * column_width_scale
                    scale_x = (column_domain[1] - column_domain[0]) * (fig_width / column_widths[c])
                    scale_y = (row_domain[1] - row_domain[0]) * (fig_height / row_heights[r])
                    _scale_translate(fig, scale_x, scale_y, column_domain[0], row_domain[0])
                merge_figure(output_figure, fig)
    if responsive:
        output_figure['config'] = {'responsive': True}
    if height:
        output_figure['layout']['height'] = height
    elif responsive:
        output_figure['layout']['autosize'] = True
    else:
        output_figure['layout']['height'] = sum(row_heights) + row_spacing * (nrows - 1)
    if width:
        output_figure['layout']['width'] = width
    elif responsive:
        output_figure['layout']['autosize'] = True
    else:
        output_figure['layout']['width'] = sum(column_widths) + column_spacing * (ncols - 1)
    if share_xaxis:
        for prop, val in output_figure['layout'].items():
            if prop.startswith('xaxis'):
                val['matches'] = 'x'
    if share_yaxis:
        for prop, val in output_figure['layout'].items():
            if prop.startswith('yaxis'):
                val['matches'] = 'y'
    return output_figure