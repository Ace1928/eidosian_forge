import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def merge_figure(fig, subfig):
    """
    Merge a sub-figure into a parent figure

    Note: This function mutates the input fig dict, but it does not mutate
    the subfig dict

    Parameters
    ----------
    fig: dict
        The plotly figure dict into which the sub figure will be merged
    subfig: dict
        The plotly figure dict that will be copied and then merged into `fig`
    """
    data = fig.setdefault('data', [])
    data.extend(copy.deepcopy(subfig.get('data', [])))
    layout = fig.setdefault('layout', {})
    merge_layout(layout, subfig.get('layout', {}))