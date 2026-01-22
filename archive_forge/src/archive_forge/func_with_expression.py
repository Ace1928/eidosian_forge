from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
def with_expression(self, key: _AttrType, expression: _ColumnExpressionArgument[Any]) -> Self:
    """Apply an ad-hoc SQL expression to a "deferred expression"
        attribute.

        This option is used in conjunction with the
        :func:`_orm.query_expression` mapper-level construct that indicates an
        attribute which should be the target of an ad-hoc SQL expression.

        E.g.::

            stmt = select(SomeClass).options(
                with_expression(SomeClass.x_y_expr, SomeClass.x + SomeClass.y)
            )

        .. versionadded:: 1.2

        :param key: Attribute to be populated

        :param expr: SQL expression to be applied to the attribute.

        .. seealso::

            :ref:`orm_queryguide_with_expression` - background and usage
            examples

        """
    expression = _orm_full_deannotate(coercions.expect(roles.LabeledColumnExprRole, expression))
    return self._set_column_strategy((key,), {'query_expression': True}, extra_criteria=(expression,))