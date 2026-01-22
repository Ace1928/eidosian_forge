import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_pandas(obj):
    """Attempt to convert pandas.NaT / pandas.NA"""
    pandas = get_module('pandas', should_load=False)
    if not pandas:
        raise NotEncodable
    if obj is pandas.NaT:
        return None
    if hasattr(pandas, 'NA') and obj is pandas.NA:
        return None
    raise NotEncodable