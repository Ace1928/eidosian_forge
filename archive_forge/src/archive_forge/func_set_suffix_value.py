import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref
import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
def set_suffix_value(self, suffix_or_name, value, expand=True):
    """Set the suffix value for this component data"""
    if isinstance(suffix_or_name, str):
        import pyomo.core.base.suffix
        for name_, suffix_ in pyomo.core.base.suffix.active_suffix_generator(self.model()):
            if suffix_or_name == name_:
                suffix_.set_value(self, value, expand=expand)
                break
    else:
        suffix_or_name.set_value(self, value, expand=expand)