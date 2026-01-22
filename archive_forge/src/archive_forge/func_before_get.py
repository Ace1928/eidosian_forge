from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def before_get(self, parser, section, option, value, vars):
    rawval = value
    depth = MAX_INTERPOLATION_DEPTH
    while depth:
        depth -= 1
        if value and '%(' in value:
            replace = functools.partial(self._interpolation_replace, parser=parser)
            value = self._KEYCRE.sub(replace, value)
            try:
                value = value % vars
            except KeyError as e:
                raise InterpolationMissingOptionError(option, section, rawval, e.args[0]) from None
        else:
            break
    if value and '%(' in value:
        raise InterpolationDepthError(option, section, rawval)
    return value