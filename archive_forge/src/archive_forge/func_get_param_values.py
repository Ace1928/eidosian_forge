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
def get_param_values(data):
    params = dict(kdims=data.kdims, vdims=data.vdims, label=data.label)
    if data.group != data.param.objects(False)['group'].default and (not isinstance(type(data).group, property)):
        params['group'] = data.group
    return params