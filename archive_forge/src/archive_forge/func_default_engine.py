import json
import decimal
import datetime
import warnings
from pathlib import Path
from plotly.io._utils import validate_coerce_fig_to_dict, validate_coerce_output_type
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
from_json_plotly requires a string or bytes argument but received value of type {typ}
@default_engine.setter
def default_engine(self, val):
    if val not in JsonConfig._valid_engines:
        raise ValueError('Supported JSON engines include {valid}\n    Received {val}'.format(valid=JsonConfig._valid_engines, val=val))
    if val == 'orjson':
        self.validate_orjson()
    self._default_engine = val