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
def recurse_data(data):
    if hasattr(data, 'to_json'):
        data = data.__dict__
    if isinstance(data, dict):
        data = dict(data)
        lower_camel_case_keys(data)
        data = {k: recurse_data(v) if k != 'data' else v for k, v in data.items()}
    elif isinstance(data, list):
        data = [recurse_data(d) for d in data]
    return data