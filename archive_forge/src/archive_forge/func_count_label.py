import collections
import functools
from typing import OrderedDict
def count_label(label):
    prev = simple_call_counter.setdefault(label, 0)
    simple_call_counter[label] = prev + 1