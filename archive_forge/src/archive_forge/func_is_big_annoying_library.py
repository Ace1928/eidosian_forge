import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def is_big_annoying_library(context):
    string_names = context.get_root_context().string_names
    if string_names is None:
        return False
    return string_names[0] in ('pandas', 'numpy', 'tensorflow', 'matplotlib')