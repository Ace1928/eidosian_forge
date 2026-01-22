from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
def generate_for_superclasses(self) -> Iterator[PathRegistry]:
    parent = self.parent
    if is_root(parent):
        yield self
        return
    if TYPE_CHECKING:
        assert isinstance(parent, AbstractEntityRegistry)
    if not parent.is_aliased_class:
        for mp_ent in parent.mapper.iterate_to_root():
            yield TokenRegistry(parent.parent[mp_ent], self.token)
    elif parent.is_aliased_class and cast('AliasedInsp[Any]', parent.entity)._is_with_polymorphic:
        yield self
        for ent in cast('AliasedInsp[Any]', parent.entity)._with_polymorphic_entities:
            yield TokenRegistry(parent.parent[ent], self.token)
    else:
        yield self