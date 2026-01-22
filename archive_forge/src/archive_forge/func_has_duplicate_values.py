from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def has_duplicate_values(self, condition_values):
    seen = set()
    for value in condition_values:
        if value.has_constant_result():
            if value.constant_result in seen:
                return True
            seen.add(value.constant_result)
        else:
            try:
                value_entry = value.entry
                if (value_entry.type.is_enum or value_entry.type.is_cpp_enum) and value_entry.enum_int_value is not None:
                    value_for_seen = value_entry.enum_int_value
                else:
                    value_for_seen = value_entry.cname
            except AttributeError:
                return True
            if value_for_seen in seen:
                return True
            seen.add(value_for_seen)
    return False