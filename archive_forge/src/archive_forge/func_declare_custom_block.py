import copy
import logging
import sys
import weakref
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory
def declare_custom_block(name, new_ctype=None):
    """Decorator to declare components for a custom block data class

    >>> @declare_custom_block(name=FooBlock)
    ... class FooBlockData(_BlockData):
    ...    # custom block data class
    ...    pass
    """

    def proc_dec(cls):
        clsbody = {'__module__': cls.__module__, '_ComponentDataClass': cls, '_default_ctype': None}
        c = type(name, (CustomBlock,), clsbody)
        if new_ctype is not None:
            if new_ctype is True:
                c._default_ctype = c
            elif type(new_ctype) is type:
                c._default_ctype = new_ctype
            else:
                raise ValueError("Expected new_ctype to be either type or 'True'; received: %s" % (new_ctype,))
        setattr(sys.modules[cls.__module__], name, c)
        setattr(cls, '_orig_name', name)
        setattr(cls, '_orig_module', cls.__module__)
        return cls
    return proc_dec