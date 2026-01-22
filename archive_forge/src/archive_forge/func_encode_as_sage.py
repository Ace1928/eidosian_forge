import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_sage(obj):
    """Attempt to convert sage.all.RR to floats and sage.all.ZZ to ints"""
    sage_all = get_module('sage.all')
    if not sage_all:
        raise NotEncodable
    if obj in sage_all.RR:
        return float(obj)
    elif obj in sage_all.ZZ:
        return int(obj)
    else:
        raise NotEncodable