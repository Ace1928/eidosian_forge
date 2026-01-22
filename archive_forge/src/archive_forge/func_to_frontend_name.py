import ast
import sys
from typing import Any
from .internal import Filters, Key
def to_frontend_name(name):
    return fe_name_map_reversed.get(name, name)