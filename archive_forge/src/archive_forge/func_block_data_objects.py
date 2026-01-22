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
def block_data_objects(self, active=None, sort=False, descend_into=True, descent_order=None):
    """Returns this block and any matching sub-blocks.

        This is roughly equivalent to

        .. code-block:: python

            iter(block for block in itertools.chain(
                 [self], self.component_data_objects(descend_into, ...))
                 if block.active == active)

        Notes
        -----
        The `self` block is *always* returned, regardless of the types
        indicated by `descend_into`.

        The active flag is enforced on *all* blocks, including `self`.

        Parameters
        ----------
        active: None or bool
            If not None, filter components by the active flag

        sort: None or bool or SortComponents
            Iterate over the components in a specified sorted order

        descend_into:  None or type or iterable
            Specifies the component types (`ctypes`) to return and to
            descend into.  If `True` or `None`, defaults to `(Block,)`.
            If `False`, only `self` is returned.

        descent_order: None or TraversalStrategy
            The strategy used to walk the block hierarchy.  Defaults to
            `TraversalStrategy.PrefixDepthFirstSearch`.

        Returns
        -------
        tuple or generator

        """
    if active is not None and self.active != active:
        return ()
    if not descend_into:
        return (self,)
    if descend_into is True:
        ctype = (Block,)
    elif isclass(descend_into):
        ctype = (descend_into,)
    else:
        ctype = descend_into
    dedup = _DeduplicateInfo()
    if descent_order is None or descent_order == TraversalStrategy.PrefixDepthFirstSearch:
        walker = self._prefix_dfs_iterator(ctype, active, sort, dedup)
    elif descent_order == TraversalStrategy.BreadthFirstSearch:
        walker = self._bfs_iterator(ctype, active, sort, dedup)
    elif descent_order == TraversalStrategy.PostfixDepthFirstSearch:
        walker = self._postfix_dfs_iterator(ctype, active, sort, dedup)
    else:
        raise RuntimeError('unrecognized traversal strategy: %s' % (descent_order,))
    return walker