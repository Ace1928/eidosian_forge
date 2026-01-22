import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def parse_kv(curr_token):
    dct = defaultdict(list)
    while curr_token.category == Pattern.KEYS:
        key = curr_token.value
        curr_token = next(tokens)
        category = curr_token.category
        if category == Pattern.REALS or category == Pattern.INTS:
            value = curr_token.value
            curr_token = next(tokens)
        elif category == Pattern.STRINGS:
            value = unescape(curr_token.value[1:-1])
            if destringizer:
                try:
                    value = destringizer(value)
                except ValueError:
                    pass
            if value == '()':
                value = ()
            if value == '[]':
                value = []
            curr_token = next(tokens)
        elif category == Pattern.DICT_START:
            curr_token, value = parse_dict(curr_token)
        elif key in ('id', 'label', 'source', 'target'):
            try:
                value = unescape(str(curr_token.value))
                if destringizer:
                    try:
                        value = destringizer(value)
                    except ValueError:
                        pass
                curr_token = next(tokens)
            except Exception:
                msg = "an int, float, string, '[' or string" + ' convertible ASCII value for node id or label'
                unexpected(curr_token, msg)
        elif curr_token.value in {'NAN', 'INF'}:
            value = float(curr_token.value)
            curr_token = next(tokens)
        else:
            unexpected(curr_token, "an int, float, string or '['")
        dct[key].append(value)

    def clean_dict_value(value):
        if not isinstance(value, list):
            return value
        if len(value) == 1:
            return value[0]
        if value[0] == LIST_START_VALUE:
            return value[1:]
        return value
    dct = {key: clean_dict_value(value) for key, value in dct.items()}
    return (curr_token, dct)