import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def get_jl_type(type_object):
    """
    Convert JS types to Julia types for the component definition
    Parameters
    ----------
    type_object: dict
        react-docgen-generated prop type dictionary
    Returns
    -------
    str
        Julia type string
    """
    js_type_name = type_object['name']
    js_to_jl_types = get_jl_prop_types(type_object=type_object)
    if js_type_name in js_to_jl_types:
        prop_type = js_to_jl_types[js_type_name]()
        return prop_type
    return ''