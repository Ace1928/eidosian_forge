from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
def replacement_traverse(obj: Optional[ExternallyTraversible], opts: Mapping[str, Any], replace: _TraverseTransformCallableType[Any]) -> Optional[ExternallyTraversible]:
    """Clone the given expression structure, allowing element
    replacement by a given replacement function.

    This function is very similar to the :func:`.visitors.cloned_traverse`
    function, except instead of being passed a dictionary of visitors, all
    elements are unconditionally passed into the given replace function.
    The replace function then has the option to return an entirely new object
    which will replace the one given.  If it returns ``None``, then the object
    is kept in place.

    The difference in usage between :func:`.visitors.cloned_traverse` and
    :func:`.visitors.replacement_traverse` is that in the former case, an
    already-cloned object is passed to the visitor function, and the visitor
    function can then manipulate the internal state of the object.
    In the case of the latter, the visitor function should only return an
    entirely different object, or do nothing.

    The use case for :func:`.visitors.replacement_traverse` is that of
    replacing a FROM clause inside of a SQL structure with a different one,
    as is a common use case within the ORM.

    """
    cloned = {}
    stop_on = {id(x) for x in opts.get('stop_on', [])}

    def deferred_copy_internals(obj: ExternallyTraversible) -> ExternallyTraversible:
        return replacement_traverse(obj, opts, replace)

    def clone(elem: ExternallyTraversible, **kw: Any) -> ExternallyTraversible:
        if id(elem) in stop_on or 'no_replacement_traverse' in elem._annotations:
            return elem
        else:
            newelem = replace(elem)
            if newelem is not None:
                stop_on.add(id(newelem))
                return newelem
            else:
                id_elem = id(elem)
                if id_elem not in cloned:
                    if 'replace' in kw:
                        newelem = kw['replace'](elem)
                        if newelem is not None:
                            cloned[id_elem] = newelem
                            return newelem
                    cloned[id_elem] = newelem = elem._clone(**kw)
                    newelem._copy_internals(clone=clone, **kw)
                return cloned[id_elem]
    if obj is not None:
        obj = clone(obj, deferred_copy_internals=deferred_copy_internals, **opts)
    clone = None
    return obj