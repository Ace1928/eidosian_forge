import collections
import copy
import itertools
import random
import re
import warnings
def pair_as_kwarg_string(key, val):
    if isinstance(val, str):
        val = val[:57] + '...' if len(val) > 60 else val
        return f"{key}='{val}'"
    return f'{key}={val}'