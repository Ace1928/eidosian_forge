from __future__ import annotations
import warnings
from collections import UserString
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
def categorical_order(vector: Series, order: list | None=None) -> list:
    """
    Return a list of unique data values using seaborn's ordering rules.

    Parameters
    ----------
    vector : Series
        Vector of "categorical" values
    order : list
        Desired order of category levels to override the order determined
        from the `data` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    if order is not None:
        return order
    if vector.dtype.name == 'category':
        order = list(vector.cat.categories)
    else:
        order = list(filter(pd.notnull, vector.unique()))
        if variable_type(pd.Series(order)) == 'numeric':
            order.sort()
    return order