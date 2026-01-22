import inspect
import logging
import sys
import textwrap
import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust
from collections.abc import Sequence
def rule_wrapper(rule, wrapping_fcn, positional_arg_map=None, map_types=None):
    """Wrap a rule with another function

    This utility method provides a way to wrap a function (rule) with
    another function while preserving the original function signature.
    This is important for rules, as the :py:func:`Initializer`
    argument processor relies on knowing the number of positional
    arguments.

    Parameters
    ----------
    rule: function
        The original rule being wrapped
    wrapping_fcn: function or Dict
        The wrapping function.  The `wrapping_fcn` will be called with
        ``(rule, *args, **kwargs)``.  For convenience, if a `dict` is
        passed as the `wrapping_fcn`, then the result of
        :py:func:`rule_result_substituter(wrapping_fcn)` is used as the
        wrapping function.
    positional_arg_map: iterable[int]
        An iterable of indices of rule positional arguments to expose in
        the wrapped function signature.  For example,
        `positional_arg_map=(2, 0)` and `rule=fcn(a, b, c)` would produce a
        wrapped function with a signature `wrapper_function(c, a)`

    """
    if isinstance(wrapping_fcn, dict):
        wrapping_fcn = rule_result_substituter(wrapping_fcn, map_types)
        if not inspect.isfunction(rule):
            return wrapping_fcn(rule)
    rule_sig = inspect.signature(rule)
    if positional_arg_map is not None:
        param = list(rule_sig.parameters.values())
        rule_sig = rule_sig.replace(parameters=(param[i] for i in positional_arg_map))
    _funcdef = _map_rule_funcdef % (str(rule_sig),)
    _env = dict(globals())
    _env.update(locals())
    exec(_funcdef, _env)
    return _env['wrapper_function']