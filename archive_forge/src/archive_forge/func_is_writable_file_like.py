import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def is_writable_file_like(obj):
    """Return whether *obj* looks like a file object with a *write* method."""
    return callable(getattr(obj, 'write', None))