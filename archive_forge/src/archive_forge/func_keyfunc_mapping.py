from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
def keyfunc_mapping(keyfunc: _F, *, ignore_unpopulated_attribute: bool=False) -> Type[KeyFuncDict[_KT, Any]]:
    """A dictionary-based collection type with arbitrary keying.

    .. versionchanged:: 2.0 Renamed :data:`.mapped_collection` to
       :func:`.keyfunc_mapping`.

    Returns a :class:`.KeyFuncDict` factory with a keying function
    generated from keyfunc, a callable that takes an entity and returns a
    key value.

    .. note:: the given keyfunc is called only once at the time that the
       target object is being added to the collection.   Changes to the
       effective value returned by the function are not tracked.


    .. seealso::

        :ref:`orm_dictionary_collection` - background on use

    :param keyfunc: a callable that will be passed the ORM-mapped instance
     which should then generate a new key to use in the dictionary.
     If the value returned is :attr:`.LoaderCallableStatus.NO_VALUE`, an error
     is raised.
    :param ignore_unpopulated_attribute:  if True, and the callable returns
     :attr:`.LoaderCallableStatus.NO_VALUE` for a particular instance, the
     operation will be silently skipped.  By default, an error is raised.

     .. versionadded:: 2.0 an error is raised by default if the callable
        being used for the dictionary key returns
        :attr:`.LoaderCallableStatus.NO_VALUE`, which in an ORM attribute
        context indicates an attribute that was never populated with any value.
        The :paramref:`_orm.mapped_collection.ignore_unpopulated_attribute`
        parameter may be set which will instead indicate that this condition
        should be ignored, and the append operation silently skipped. This is
        in contrast to the behavior of the 1.x series which would erroneously
        populate the value in the dictionary with an arbitrary key value of
        ``None``.


    """
    return _mapped_collection_cls(keyfunc, ignore_unpopulated_attribute=ignore_unpopulated_attribute)