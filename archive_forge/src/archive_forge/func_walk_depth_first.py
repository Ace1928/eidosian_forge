import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def walk_depth_first(name):
    stack = [name]
    while stack:
        name = stack.pop()
        if name in levels_by_name:
            continue
        if name not in graph or not graph[name]:
            level = 0
            add_level_to_name(name, level)
            continue
        children = graph[name]
        children_not_calculated = [child for child in children if child not in levels_by_name]
        if children_not_calculated:
            stack.append(name)
            stack.extend(children_not_calculated)
            continue
        level = 1 + max((levels_by_name[lname] for lname in children))
        add_level_to_name(name, level)