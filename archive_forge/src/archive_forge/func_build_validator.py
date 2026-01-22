import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
@staticmethod
def build_validator(validator_info, plotly_name, parent_name):
    datatype = validator_info['valType']
    validator_classname = datatype.title().replace('_', '') + 'Validator'
    validator_class = eval(validator_classname)
    kwargs = {k: validator_info[k] for k in validator_info if k not in ['valType', 'description', 'role']}
    return validator_class(plotly_name=plotly_name, parent_name=parent_name, **kwargs)