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
def load_only(self, *attrs: _AttrType, raiseload: bool=False) -> Self:
    """Indicate that for a particular entity, only the given list
        of column-based attribute names should be loaded; all others will be
        deferred.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        Example - given a class ``User``, load only the ``name`` and
        ``fullname`` attributes::

            session.query(User).options(load_only(User.name, User.fullname))

        Example - given a relationship ``User.addresses -> Address``, specify
        subquery loading for the ``User.addresses`` collection, but on each
        ``Address`` object load only the ``email_address`` attribute::

            session.query(User).options(
                subqueryload(User.addresses).load_only(Address.email_address)
            )

        For a statement that has multiple entities,
        the lead entity can be
        specifically referred to using the :class:`_orm.Load` constructor::

            stmt = (
                select(User, Address)
                .join(User.addresses)
                .options(
                    Load(User).load_only(User.name, User.fullname),
                    Load(Address).load_only(Address.email_address),
                )
            )

        When used together with the
        :ref:`populate_existing <orm_queryguide_populate_existing>`
        execution option only the attributes listed will be refreshed.

        :param \\*attrs: Attributes to be loaded, all others will be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when a deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

         .. versionadded:: 2.0

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

        :param \\*attrs: Attributes to be loaded, all others will be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when a deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

         .. versionadded:: 2.0

        """
    cloned = self._set_column_strategy(attrs, {'deferred': False, 'instrument': True})
    wildcard_strategy = {'deferred': True, 'instrument': True}
    if raiseload:
        wildcard_strategy['raiseload'] = True
    cloned = cloned._set_column_strategy(('*',), wildcard_strategy)
    return cloned