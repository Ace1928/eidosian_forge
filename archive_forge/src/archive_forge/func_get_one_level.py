from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def get_one_level(key_list, level, d):
    if not list_ok:
        assert isinstance(d, dict)
    if level >= len(key_list):
        if level > len(key_list):
            raise IndexError
        return d[key_list[level - 1]]
    return get_one_level(key_list, level + 1, d[key_list[level - 1]])