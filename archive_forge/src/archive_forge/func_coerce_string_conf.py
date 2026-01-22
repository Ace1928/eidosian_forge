import abc
import collections
import re
import threading
from typing import MutableMapping
from typing import MutableSet
import stevedore
def coerce_string_conf(d):
    result = {}
    for k, v in d.items():
        if not isinstance(v, str):
            result[k] = v
            continue
        v = v.strip()
        if re.match('^[-+]?\\d+$', v):
            result[k] = int(v)
        elif re.match('^[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?$', v):
            result[k] = float(v)
        elif v.lower() in ('false', 'true'):
            result[k] = v.lower() == 'true'
        elif v == 'None':
            result[k] = None
        else:
            result[k] = v
    return result