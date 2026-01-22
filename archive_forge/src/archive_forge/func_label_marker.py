from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def label_marker(i):
    if i[1].offset in self.labels:
        return '>'
    else:
        return ' '