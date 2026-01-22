import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def group_data(data, groupby_column_name, use_mean=None):
    """
    Group data by scenario

    Parameters
    ----------
    data: DataFrame
        Data
    groupby_column_name: strings
        Name of data column which contains scenario numbers
    use_mean: list of column names or None, optional
        Name of data columns which should be reduced to a single value per
        scenario by taking the mean

    Returns
    ----------
    grouped_data: list of dictionaries
        Grouped data
    """
    if use_mean is None:
        use_mean_list = []
    else:
        use_mean_list = use_mean
    grouped_data = []
    for exp_num, group in data.groupby(data[groupby_column_name]):
        d = {}
        for col in group.columns:
            if col in use_mean_list:
                d[col] = group[col].mean()
            else:
                d[col] = list(group[col])
        grouped_data.append(d)
    return grouped_data