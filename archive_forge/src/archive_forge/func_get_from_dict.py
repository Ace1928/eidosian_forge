from __future__ import absolute_import, division, print_function
import ast
import json
import operator
import re
import socket
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from itertools import chain
from ansible.module_utils import basic
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems, string_types
def get_from_dict(data_dict, keypath):
    """get from dictionary"""
    map_list = keypath.split('.')
    try:
        return reduce(operator.getitem, map_list, data_dict)
    except KeyError:
        return None