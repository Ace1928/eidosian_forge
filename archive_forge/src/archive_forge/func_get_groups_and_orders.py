import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def get_groups_and_orders(args, grouper):
    """
    `orders` is the user-supplied ordering with the remaining data-frame-supplied
    ordering appended if the column is used for grouping. It includes anything the user
    gave, for any variable, including values not present in the dataset. It's a dict
    where the keys are e.g. "x" or "color"

    `groups` is the dicts of groups, ordered by the order above. Its keys are
    tuples like [("value1", ""), ("value2", "")] where each tuple contains the name
    of a single dimension-group
    """
    orders = {} if 'category_orders' not in args else args['category_orders'].copy()
    single_group_name = []
    unique_cache = dict()
    for col in grouper:
        if col == one_group:
            single_group_name.append('')
        else:
            if col not in unique_cache:
                unique_cache[col] = list(args['data_frame'][col].unique())
            uniques = unique_cache[col]
            if len(uniques) == 1:
                single_group_name.append(uniques[0])
            if col not in orders:
                orders[col] = uniques
            else:
                orders[col] = list(OrderedDict.fromkeys(list(orders[col]) + uniques))
    df = args['data_frame']
    if len(single_group_name) == len(grouper):
        groups = {tuple(single_group_name): df}
    else:
        required_grouper = [g for g in grouper if g != one_group]
        grouped = df.groupby(required_grouper, sort=False, observed=True)
        group_indices = grouped.indices
        sorted_group_names = [g if len(required_grouper) != 1 else (g,) for g in group_indices]
        for i, col in reversed(list(enumerate(required_grouper))):
            sorted_group_names = sorted(sorted_group_names, key=lambda g: orders[col].index(g[i]) if g[i] in orders[col] else -1)
        full_sorted_group_names = [list(t) for t in sorted_group_names]
        for i, col in enumerate(grouper):
            if col == one_group:
                for g in full_sorted_group_names:
                    g.insert(i, '')
        full_sorted_group_names = [tuple(g) for g in full_sorted_group_names]
        groups = {}
        for sf, s in zip(full_sorted_group_names, sorted_group_names):
            if len(s) > 1:
                groups[sf] = grouped.get_group(s)
            elif pandas_2_2_0:
                groups[sf] = grouped.get_group((s[0],))
            else:
                groups[sf] = grouped.get_group(s[0])
    return (groups, orders)