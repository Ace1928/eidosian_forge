from __future__ import annotations
import json
import sys
from collections import defaultdict
from typing import (
import numpy as np
import param
from bokeh.core.serialization import Serializer
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import is_dataframe, lazy_load
from .base import ModelPane
def lower_camel_case_keys(attrs):
    """
    Makes all the keys in a dictionary camel-cased and lower-case

    Parameters
    ----------
    attrs : dict
        Dictionary for which all the keys should be converted to camel-case
    """
    for snake_key in list(attrs.keys()):
        if '_' not in snake_key:
            continue
        camel_key = lower_first_letter(to_camel_case(snake_key))
        attrs[camel_key] = attrs.pop(snake_key)