import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
def process_setarg(arg):
    if isinstance(arg, _SetDataBase):
        if getattr(arg, '_parent', None) is not None or getattr(arg, '_anonymous_sets', None) is GlobalSetBase or arg.parent_component()._parent is not None:
            return (arg, None)
        _anonymous = ComponentSet((arg,))
        if getattr(arg, '_anonymous_sets', None) is not None:
            _anonymous.update(arg._anonymous_sets)
        return (arg, _anonymous)
    elif isinstance(arg, _ComponentBase):
        if isinstance(arg, IndexedComponent) and arg.is_indexed():
            raise TypeError('Cannot apply a Set operator to an indexed %s component (%s)' % (arg.ctype.__name__, arg.name))
        if isinstance(arg, Component):
            raise TypeError('Cannot apply a Set operator to a non-Set %s component (%s)' % (arg.__class__.__name__, arg.name))
        if isinstance(arg, ComponentData):
            raise TypeError('Cannot apply a Set operator to a non-Set component data (%s)' % (arg.name,))
    if hasattr(arg, 'set_options'):
        deprecation_warning('The set_options set attribute is deprecated.  Please explicitly construct complex sets', version='5.7.3')
        args = arg.set_options
        args.setdefault('initialize', arg)
        args.setdefault('ordered', type(arg) not in Set._UnorderedInitializers)
        ans = Set(**args)
        _init = args['initialize']
        if not (inspect.isgenerator(_init) or inspect.isfunction(_init) or (isinstance(_init, ComponentData) and (not _init.parent_component().is_constructed()))):
            ans.construct()
        return process_setarg(ans)
    _defer_construct = False
    if not hasattr(arg, '__contains__'):
        if inspect.isgenerator(arg):
            _ordered = True
            _defer_construct = True
        elif inspect.isfunction(arg):
            _ordered = True
            _defer_construct = True
        else:
            raise TypeError("Cannot create a Set from data that does not support __contains__.  Expected set-like object supporting collections.abc.Collection interface, but received '%s'." % (type(arg).__name__,))
    elif arg.__class__ is type:
        return process_setarg(arg())
    else:
        arg = SetOf(arg)
        _ordered = arg.isordered()
    ans = Set(initialize=arg, ordered=_ordered)
    if not _defer_construct:
        ans.construct()
    _anonymous = ComponentSet((ans,))
    if getattr(ans, '_anonymous_sets', None) is not None:
        _anonymous.update(_anonymous_sets)
    return (ans, _anonymous)