import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace('function', info={'info': 'some_info'})
def traced_func(i):
    return i