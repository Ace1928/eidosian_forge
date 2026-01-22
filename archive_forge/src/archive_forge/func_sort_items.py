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
def sort_items(items, sort_str, sort_type=None):
    """Sort items based on sort keys and sort directions given by sort_str.

    :param items: a list or generator object of items
    :param sort_str: a string defining the sort rules, the format is
        '<key1>:[direction1],<key2>:[direction2]...', direction can be 'asc'
        for ascending or 'desc' for descending, if direction is not given,
        it's ascending by default
    :return: sorted items
    """
    if not sort_str:
        return items
    items = list(items)
    sort_keys = sort_str.strip().split(',')
    for sort_key in reversed(sort_keys):
        reverse = False
        if ':' in sort_key:
            sort_key, direction = sort_key.split(':', 1)
            if not sort_key:
                msg = _("'<empty string>'' is not a valid sort key")
                raise exceptions.CommandError(msg)
            if direction not in ['asc', 'desc']:
                if not direction:
                    direction = '<empty string>'
                msg = _("'%(direction)s' is not a valid sort direction for sort key %(sort_key)s, use 'asc' or 'desc' instead")
                raise exceptions.CommandError(msg % {'direction': direction, 'sort_key': sort_key})
            if direction == 'desc':
                reverse = True

        def f(x):
            item = get_field(x, sort_key)
            if sort_type and (not isinstance(item, sort_type)):
                try:
                    item = sort_type(item)
                except Exception:
                    item = sort_type()
            return item
        items.sort(key=f, reverse=reverse)
    return items