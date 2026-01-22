from __future__ import print_function
import os
import io
import time
import functools
import collections
import collections.abc
import numpy as np
import requests
import IPython
def nested_setitem(obj, dotted_name, value):
    items = dotted_name.split('.')
    for item in items[:-1]:
        if item not in obj:
            obj[item] = {}
        obj = obj[item]
    obj[items[-1]] = value