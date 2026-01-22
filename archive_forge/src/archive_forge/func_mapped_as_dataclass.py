from __future__ import annotations
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import instrumentation
from . import interfaces
from . import mapperlib
from ._orm_constructors import composite
from ._orm_constructors import deferred
from ._orm_constructors import mapped_column
from ._orm_constructors import relationship
from ._orm_constructors import synonym
from .attributes import InstrumentedAttribute
from .base import _inspect_mapped_class
from .base import _is_mapped_class
from .base import Mapped
from .base import ORMDescriptor
from .decl_base import _add_attribute
from .decl_base import _as_declarative
from .decl_base import _ClassScanMapperConfig
from .decl_base import _declarative_constructor
from .decl_base import _DeferredMapperConfig
from .decl_base import _del_attribute
from .decl_base import _mapper
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .descriptor_props import Synonym as _orm_synonym
from .mapper import Mapper
from .properties import MappedColumn
from .relationships import RelationshipProperty
from .state import InstanceState
from .. import exc
from .. import inspection
from .. import util
from ..sql import sqltypes
from ..sql.base import _NoArg
from ..sql.elements import SQLCoreOperations
from ..sql.schema import MetaData
from ..sql.selectable import FromClause
from ..util import hybridmethod
from ..util import hybridproperty
from ..util import typing as compat_typing
from ..util.typing import CallableReference
from ..util.typing import flatten_newtype
from ..util.typing import is_generic
from ..util.typing import is_literal
from ..util.typing import is_newtype
from ..util.typing import is_pep695
from ..util.typing import Literal
from ..util.typing import Self
def mapped_as_dataclass(self, __cls: Optional[Type[_O]]=None, *, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, eq: Union[_NoArg, bool]=_NoArg.NO_ARG, order: Union[_NoArg, bool]=_NoArg.NO_ARG, unsafe_hash: Union[_NoArg, bool]=_NoArg.NO_ARG, match_args: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]=_NoArg.NO_ARG) -> Union[Type[_O], Callable[[Type[_O]], Type[_O]]]:
    """Class decorator that will apply the Declarative mapping process
        to a given class, and additionally convert the class to be a
        Python dataclass.

        .. seealso::

            :ref:`orm_declarative_native_dataclasses` - complete background
            on SQLAlchemy native dataclass mapping


        .. versionadded:: 2.0


        """

    def decorate(cls: Type[_O]) -> Type[_O]:
        setattr(cls, '_sa_apply_dc_transforms', {'init': init, 'repr': repr, 'eq': eq, 'order': order, 'unsafe_hash': unsafe_hash, 'match_args': match_args, 'kw_only': kw_only, 'dataclass_callable': dataclass_callable})
        _as_declarative(self, cls, cls.__dict__)
        return cls
    if __cls:
        return decorate(__cls)
    else:
        return decorate