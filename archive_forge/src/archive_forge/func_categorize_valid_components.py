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
def categorize_valid_components(model, active=True, sort=None, valid=set(), targets=set()):
    """Walk model and check for valid component types

    This routine will walk the model and check all component types.
    Components types in the `valid` set are ignored, blocks with
    components in the `targets` set are collected, and all other
    component types are added to a dictionary of `unrecognized`
    components.

    A Component type may not appear in both `valid` and `targets` sets.

    Parameters
    ----------
    model: _BlockData
        The model tree to walk

    active: True or None
        If True, only unrecognized active components are returned in the
        `uncategorized` dictionary.  Also, if True, only active Blocks
        are descended into.

    sort: bool or SortComponents
        The sorting flag to pass to the block walkers

    valid: Set[type]
        The set of "valid" component types.  These are ignored by the
        categorizer.

    targets: Set[type]
        The set of component types to "collect".  Blocks with components
        in the `targets` set will be returned in the `component_map`

    Returns
    -------
    component_map: Dict[type, List[_BlockData]]
        A dict mapping component type to a list of block data
        objects that contain declared component of that type.

    unrecognized: Dict[type, List[ComponentData]]
        A dict mapping unrecognized component types to a (non-empty)
        list of component data objects found on the model.

    """
    assert active in (True, None)
    if any((ctype in valid for ctype in targets)):
        ctypes = list(filter(valid.__contains__, targets))
        raise DeveloperError(f'categorize_valid_components: Cannot have component type {ctypes} in both the `valid` and `targets` sets')
    unrecognized = {}
    component_map = {k: [] for k in targets}
    for block in model.block_data_objects(active=active, descend_into=True, sort=sort):
        local_ctypes = block.collect_ctypes(active=None, descend_into=False)
        for ctype in local_ctypes:
            if ctype in kernel.base._kernel_ctype_backmap:
                ctype = kernel.base._kernel_ctype_backmap[ctype]
            if ctype in valid:
                continue
            if ctype in targets:
                component_map[ctype].append(block)
                continue
            if active and (not issubclass(ctype, ActiveComponent)) and (not issubclass(ctype, kernel.base.ICategorizedObject)):
                continue
            if ctype not in unrecognized:
                unrecognized[ctype] = []
            unrecognized[ctype].extend(block.component_data_objects(ctype=ctype, active=active, descend_into=False, sort=SortComponents.unsorted))
    return (component_map, {k: v for k, v in unrecognized.items() if v})