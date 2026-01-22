from __future__ import annotations
import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def visit_multi_list(self, attrname: str, obj: Any, parent: Any, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
    return (attrname, tuple((elem._gen_cache_key(anon_map, bindparams) if isinstance(elem, HasCacheKey) else elem for elem in obj)))