from __future__ import print_function
import sys
import os
import types
import traceback
import pprint
import argparse
from srsly.ruamel_yaml.compat import PY3
def find_test_functions(collections):
    if not isinstance(collections, list):
        collections = [collections]
    functions = []
    for collection in collections:
        if not isinstance(collection, dict):
            collection = vars(collection)
        for key in sorted(collection):
            value = collection[key]
            if isinstance(value, types.FunctionType) and hasattr(value, 'unittest'):
                functions.append(value)
    return functions