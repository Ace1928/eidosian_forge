import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def doSubs(s):
    return functools.reduce(doSub, subs, s)