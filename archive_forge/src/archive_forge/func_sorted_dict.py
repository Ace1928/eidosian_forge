from __future__ import absolute_import, division, print_function
import json
import hashlib
def sorted_dict(unsorted_dict):
    result = OrderedDict()
    for k, v in sorted(unsorted_dict.items()):
        if isinstance(v, dict):
            v = sorted_dict(v)
        result[k] = v
    return result