import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def get_str_or_none(value):
    return _get_safe_value_or_none(value, str)