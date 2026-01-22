from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import mapperlib as mapperlib
from ._typing import _O
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .interfaces import _AttributeOptions
from .properties import MappedColumn
from .properties import MappedSQLExpression
from .query import AliasOption
from .relationships import _RelationshipArgumentType
from .relationships import _RelationshipDeclared
from .relationships import _RelationshipSecondaryArgument
from .relationships import RelationshipProperty
from .session import Session
from .util import _ORMJoin
from .util import AliasedClass
from .util import AliasedInsp
from .util import LoaderCriteriaOption
from .. import sql
from .. import util
from ..exc import InvalidRequestError
from ..sql._typing import _no_kw
from ..sql.base import _NoArg
from ..sql.base import SchemaEventTarget
from ..sql.schema import _InsertSentinelColumnDefault
from ..sql.schema import SchemaConst
from ..sql.selectable import FromClause
from ..util.typing import Annotated
from ..util.typing import Literal
@util.deprecated_params(**{arg: ('2.0', f'The :paramref:`_orm.column_property.{arg}` parameter is deprecated for :func:`_orm.column_property`.  This parameter applies to a writeable-attribute in a Declarative Dataclasses configuration only, and :func:`_orm.column_property` is treated as a read-only attribute in this context.') for arg in ('init', 'kw_only', 'default', 'default_factory')})
def column_property(column: _ORMColumnExprArgument[_T], *additional_columns: _ORMColumnExprArgument[Any], group: Optional[str]=None, deferred: bool=False, raiseload: bool=False, comparator_factory: Optional[Type[PropComparator[_T]]]=None, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, default: Optional[Any]=_NoArg.NO_ARG, default_factory: Union[_NoArg, Callable[[], _T]]=_NoArg.NO_ARG, compare: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, active_history: bool=False, expire_on_flush: bool=True, info: Optional[_InfoType]=None, doc: Optional[str]=None) -> MappedSQLExpression[_T]:
    """Provide a column-level property for use with a mapping.

    With Declarative mappings, :func:`_orm.column_property` is used to
    map read-only SQL expressions to a mapped class.

    When using Imperative mappings, :func:`_orm.column_property` also
    takes on the role of mapping table columns with additional features.
    When using fully Declarative mappings, the :func:`_orm.mapped_column`
    construct should be used for this purpose.

    With Declarative Dataclass mappings, :func:`_orm.column_property`
    is considered to be **read only**, and will not be included in the
    Dataclass ``__init__()`` constructor.

    The :func:`_orm.column_property` function returns an instance of
    :class:`.ColumnProperty`.

    .. seealso::

        :ref:`mapper_column_property_sql_expressions` - general use of
        :func:`_orm.column_property` to map SQL expressions

        :ref:`orm_imperative_table_column_options` - usage of
        :func:`_orm.column_property` with Imperative Table mappings to apply
        additional options to a plain :class:`_schema.Column` object

    :param \\*cols:
        list of Column objects to be mapped.

    :param active_history=False:

        Used only for Imperative Table mappings, or legacy-style Declarative
        mappings (i.e. which have not been upgraded to
        :func:`_orm.mapped_column`), for column-based attributes that are
        expected to be writeable; use :func:`_orm.mapped_column` with
        :paramref:`_orm.mapped_column.active_history` for Declarative mappings.
        See that parameter for functional details.

    :param comparator_factory: a class which extends
        :class:`.ColumnProperty.Comparator` which provides custom SQL
        clause generation for comparison operations.

    :param group:
        a group name for this property when marked as deferred.

    :param deferred:
        when True, the column property is "deferred", meaning that
        it does not load immediately, and is instead loaded when the
        attribute is first accessed on an instance.  See also
        :func:`~sqlalchemy.orm.deferred`.

    :param doc:
        optional string that will be applied as the doc on the
        class-bound descriptor.

    :param expire_on_flush=True:
        Disable expiry on flush.   A column_property() which refers
        to a SQL expression (and not a single table-bound column)
        is considered to be a "read only" property; populating it
        has no effect on the state of data, and it can only return
        database state.   For this reason a column_property()'s value
        is expired whenever the parent object is involved in a
        flush, that is, has any kind of "dirty" state within a flush.
        Setting this parameter to ``False`` will have the effect of
        leaving any existing value present after the flush proceeds.
        Note that the :class:`.Session` with default expiration
        settings still expires
        all attributes after a :meth:`.Session.commit` call, however.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.MapperProperty.info` attribute of this object.

    :param raiseload: if True, indicates the column should raise an error
        when undeferred, rather than loading the value.  This can be
        altered at query time by using the :func:`.deferred` option with
        raiseload=False.

        .. versionadded:: 1.4

        .. seealso::

            :ref:`orm_queryguide_deferred_raiseload`

    :param init:

    :param default:

    :param default_factory:

    :param kw_only:

    """
    return MappedSQLExpression(column, *additional_columns, attribute_options=_AttributeOptions(False if init is _NoArg.NO_ARG else init, repr, default, default_factory, compare, kw_only), group=group, deferred=deferred, raiseload=raiseload, comparator_factory=comparator_factory, active_history=active_history, expire_on_flush=expire_on_flush, info=info, doc=doc, _assume_readonly_dc_attributes=True)