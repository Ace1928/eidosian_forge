from functools import lru_cache, partial
import re
from pyparsing import (
from matplotlib import _api
def comma_separated(elem):
    return elem + ZeroOrMore(Suppress(',') + elem)