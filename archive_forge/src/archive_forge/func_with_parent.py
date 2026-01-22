from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from . import util as orm_util
from ._typing import _O
from .base import _assertions
from .context import _column_descriptions
from .context import _determine_last_joined_entity
from .context import _legacy_filter_by_entity_zero
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .util import AliasedClass
from .util import object_mapper
from .util import with_parent
from .. import exc as sa_exc
from .. import inspect
from .. import inspection
from .. import log
from .. import sql
from .. import util
from ..engine import Result
from ..engine import Row
from ..event import dispatcher
from ..event import EventTarget
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import Select
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _FromClauseArgument
from ..sql._typing import _TP
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import _entity_namespace_key
from ..sql.base import _generative
from ..sql.base import _NoArg
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.elements import BooleanClauseList
from ..sql.expression import Exists
from ..sql.selectable import _MemoizedSelectEntities
from ..sql.selectable import _SelectFromElements
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import HasHints
from ..sql.selectable import HasPrefixes
from ..sql.selectable import HasSuffixes
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectLabelStyle
from ..util.typing import Literal
from ..util.typing import Self
@util.became_legacy_20(':meth:`_orm.Query.with_parent`', alternative='Use the :func:`_orm.with_parent` standalone construct.')
@util.preload_module('sqlalchemy.orm.relationships')
def with_parent(self, instance: object, property: Optional[attributes.QueryableAttribute[Any]]=None, from_entity: Optional[_ExternalEntityType[Any]]=None) -> Self:
    """Add filtering criterion that relates the given instance
        to a child object or collection, using its attribute state
        as well as an established :func:`_orm.relationship()`
        configuration.

        The method uses the :func:`.with_parent` function to generate
        the clause, the result of which is passed to
        :meth:`_query.Query.filter`.

        Parameters are the same as :func:`.with_parent`, with the exception
        that the given property can be None, in which case a search is
        performed against this :class:`_query.Query` object's target mapper.

        :param instance:
          An instance which has some :func:`_orm.relationship`.

        :param property:
          Class bound attribute which indicates
          what relationship from the instance should be used to reconcile the
          parent/child relationship.

        :param from_entity:
          Entity in which to consider as the left side.  This defaults to the
          "zero" entity of the :class:`_query.Query` itself.

        """
    relationships = util.preloaded.orm_relationships
    if from_entity:
        entity_zero = inspect(from_entity)
    else:
        entity_zero = _legacy_filter_by_entity_zero(self)
    if property is None:
        mapper = object_mapper(instance)
        for prop in mapper.iterate_properties:
            if isinstance(prop, relationships.RelationshipProperty) and prop.mapper is entity_zero.mapper:
                property = prop
                break
        else:
            raise sa_exc.InvalidRequestError("Could not locate a property which relates instances of class '%s' to instances of class '%s'" % (entity_zero.mapper.class_.__name__, instance.__class__.__name__))
    return self.filter(with_parent(instance, property, entity_zero.entity))