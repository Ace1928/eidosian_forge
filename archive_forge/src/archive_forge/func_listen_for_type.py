from within the mutable extension::
from __future__ import annotations
from collections import defaultdict
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from weakref import WeakKeyDictionary
from .. import event
from .. import inspect
from .. import types
from .. import util
from ..orm import Mapper
from ..orm._typing import _ExternalEntityType
from ..orm._typing import _O
from ..orm._typing import _T
from ..orm.attributes import AttributeEventToken
from ..orm.attributes import flag_modified
from ..orm.attributes import InstrumentedAttribute
from ..orm.attributes import QueryableAttribute
from ..orm.context import QueryContext
from ..orm.decl_api import DeclarativeAttributeIntercept
from ..orm.state import InstanceState
from ..orm.unitofwork import UOWTransaction
from ..sql.base import SchemaEventTarget
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import memoized_property
from ..util.typing import SupportsIndex
from ..util.typing import TypeGuard
def listen_for_type(mapper: Mapper[_T], class_: Union[DeclarativeAttributeIntercept, type]) -> None:
    if mapper.non_primary:
        return
    _APPLIED_KEY = '_ext_mutable_listener_applied'
    for prop in mapper.column_attrs:
        if isinstance(prop.expression, Column) and (schema_event_check and prop.expression.info.get('_ext_mutable_orig_type') is sqltype or prop.expression.type is sqltype):
            if not prop.expression.info.get(_APPLIED_KEY, False):
                prop.expression.info[_APPLIED_KEY] = True
                cls.associate_with_attribute(getattr(class_, prop.key))