import collections
import enum
import functools
import itertools
import logging
import operator
import sys
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel
def initialize_var_map_from_column_order(model, config, var_map):
    column_order = config.column_order
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)
    if column_order is None or column_order.__class__ is bool:
        if not column_order:
            column_order = None
    elif isinstance(column_order, ComponentMap):
        column_order = sorted(column_order, key=column_order.__getitem__)
    if column_order == True:
        column_order = model.component_data_objects(Var, descend_into=True, sort=sorter)
    elif config.file_determinism > FileDeterminism.ORDERED:
        var_objs = model.component_data_objects(Var, descend_into=True, sort=sorter)
        if column_order is None:
            column_order = var_objs
        else:
            column_order = itertools.chain(column_order, var_objs)
    if column_order is not None:
        fill_in = ComponentSet()
        for var in column_order:
            if var.is_indexed():
                for _v in var.values(sorter):
                    if not _v.fixed:
                        var_map[id(_v)] = _v
            elif not var.fixed:
                pc = var.parent_component()
                if pc is not var and pc not in fill_in:
                    fill_in.add(pc)
                var_map[id(var)] = var
        for pc in fill_in:
            for _v in pc.values(sorter):
                if not _v.fixed:
                    var_map[id(_v)] = _v
    return var_map