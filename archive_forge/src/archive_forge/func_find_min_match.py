import copy
import functools
import getpass
import logging
import os
import time
import warnings
from cliff import columns as cliff_columns
from oslo_utils import importutils
from osc_lib import exceptions
from osc_lib.i18n import _
def find_min_match(items, sort_attr, **kwargs):
    """Find all resources meeting the given minimum constraints

    :param items: A List of objects to consider
    :param sort_attr: Attribute to sort the resulting list
    :param kwargs: A dict of attributes and their minimum values
    :rtype: A list of resources osrted by sort_attr that meet the minimums
    """

    def minimum_pieces_of_flair(item):
        """Find lowest value greater than the minumum"""
        result = True
        for k in kwargs:
            result = result and kwargs[k] <= get_field(item, k)
        return result
    return sort_items(filter(minimum_pieces_of_flair, items), sort_attr)