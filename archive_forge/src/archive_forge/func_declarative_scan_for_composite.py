from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import strategy_options
from .base import _DeclarativeMapped
from .base import class_mapper
from .descriptor_props import CompositeProperty
from .descriptor_props import ConcreteInheritedProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .interfaces import StrategizedProperty
from .relationships import RelationshipProperty
from .util import de_stringify_annotation
from .util import de_stringify_union_elements
from .. import exc as sa_exc
from .. import ForeignKey
from .. import log
from .. import util
from ..sql import coercions
from ..sql import roles
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import SchemaConst
from ..sql.type_api import TypeEngine
from ..util.typing import de_optionalize_union_types
from ..util.typing import is_fwd_ref
from ..util.typing import is_optional_union
from ..util.typing import is_pep593
from ..util.typing import is_pep695
from ..util.typing import is_union
from ..util.typing import Self
from ..util.typing import typing_get_args
@util.preload_module('sqlalchemy.orm.decl_base')
def declarative_scan_for_composite(self, registry: _RegistryType, cls: Type[Any], originating_module: Optional[str], key: str, param_name: str, param_annotation: _AnnotationScanType) -> None:
    decl_base = util.preloaded.orm_decl_base
    decl_base._undefer_column_name(param_name, self.column)
    self._init_column_for_annotation(cls, registry, param_annotation, originating_module)