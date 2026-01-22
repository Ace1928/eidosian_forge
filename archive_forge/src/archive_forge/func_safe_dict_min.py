import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
def safe_dict_min(key, data):
    """Safely find the minimum for a given key in a list of dict objects.

    This will find the minimum integer value for specific dictionary key
    across a list of dictionaries. The values for the given key MUST be
    integers, or string representations of an integer.

    The dictionary key does not have to be present in all (or any)
    of the elements/dicts within the data set.

    :param string key: The dictionary key to search for the minimum value.
    :param list data: List of dicts to use for the data set.

    :returns: None if the field was not found in any elements, or
        the minimum value for the field otherwise.
    """
    min_value = None
    for d in data:
        if key in d and d[key] is not None:
            try:
                val = int(d[key])
            except ValueError:
                raise exceptions.SDKException('Search for minimum value failed. Value for {key} is not an integer: {value}'.format(key=key, value=d[key]))
            if min_value is None or val < min_value:
                min_value = val
    return min_value