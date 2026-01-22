import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def format_methods(r):
    if r.conditions:
        method = r.conditions.get('method', '')
        return type(method) is str and method or ', '.join(method)
    else:
        return ''