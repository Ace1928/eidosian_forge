import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace('hide_args', hide_args=True)
def trace_hide_args_func(a, i=10):
    return (a, i)