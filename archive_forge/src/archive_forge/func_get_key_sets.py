from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_key_sets(dict_conf):
    key_set = set(dict_conf.keys())
    trival_key_set = set()
    dict_list_key_set = set()
    for key in key_set:
        if dict_conf[key] not in [None, [], {}]:
            if isinstance(dict_conf[key], (list, dict)):
                dict_list_key_set.add(key)
            else:
                trival_key_set.add(key)
    return (trival_key_set, dict_list_key_set)