from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def sort_models(models):
    models = set(models)
    seen = set()
    ordering = []

    def dfs(model):
        if model in models and model not in seen:
            seen.add(model)
            for foreign_key, rel_model in model._meta.refs.items():
                if not foreign_key.deferred:
                    dfs(rel_model)
            if model._meta.depends_on:
                for dependency in model._meta.depends_on:
                    dfs(dependency)
            ordering.append(model)
    names = lambda m: (m._meta.name, m._meta.table_name)
    for m in sorted(models, key=names):
        dfs(m)
    return ordering