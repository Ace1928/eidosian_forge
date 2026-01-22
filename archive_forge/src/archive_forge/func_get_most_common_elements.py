from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_most_common_elements(iterator):
    """Returns a generator containing a descending list of most common elements."""
    if not isinstance(iterator, list):
        raise TypeError('iterator must be a list.')
    grouped = [(key, len(list(group))) for key, group in groupby(sorted(iterator))]
    return sorted(grouped, key=lambda x: x[1], reverse=True)