import re
import warnings
from numbers import Number
from pathlib import Path
from typing import Dict
import numpy as np
from imageio.core.legacy_plugin_wrapper import LegacyPlugin
from imageio.core.util import Array
from imageio.core.v3_plugin_api import PluginV3
from . import formats
from .config import known_extensions, known_plugins
from .core import RETURN_BYTES
from .core.imopen import imopen
def to_nbytes(arg, default=None):
    if not arg:
        arg = float('inf')
    if arg is True:
        arg = default
    if isinstance(arg, Number):
        return arg
    match = mem_re.match(arg)
    if match is None:
        raise ValueError('Memory size could not be parsed (is your capitalisation correct?): {}'.format(arg))
    num, unit = match.groups()
    try:
        return float(num) * sizes[unit]
    except KeyError:
        raise ValueError('Memory size unit not recognised (is your capitalisation correct?): {}'.format(unit))