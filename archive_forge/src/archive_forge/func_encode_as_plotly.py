import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_plotly(obj):
    """Attempt to use a builtin `to_plotly_json` method."""
    try:
        return obj.to_plotly_json()
    except AttributeError:
        raise NotEncodable