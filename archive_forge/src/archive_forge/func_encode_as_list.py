import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_list(obj):
    """Attempt to use `tolist` method to convert to normal Python list."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        raise NotEncodable