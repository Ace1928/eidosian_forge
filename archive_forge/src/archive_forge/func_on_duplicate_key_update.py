from __future__ import annotations
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from ... import exc
from ... import util
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import KeyedColumnElement
from ...sql.expression import alias
from ...sql.selectable import NamedFromClause
from ...util.typing import Self
@_generative
@_exclusive_against('_post_values_clause', msgs={'_post_values_clause': 'This Insert construct already has an ON DUPLICATE KEY clause present'})
def on_duplicate_key_update(self, *args: _UpdateArg, **kw: Any) -> Self:
    """
        Specifies the ON DUPLICATE KEY UPDATE clause.

        :param \\**kw:  Column keys linked to UPDATE values.  The
         values may be any SQL expression or supported literal Python
         values.

        .. warning:: This dictionary does **not** take into account
           Python-specified default UPDATE values or generation functions,
           e.g. those specified using :paramref:`_schema.Column.onupdate`.
           These values will not be exercised for an ON DUPLICATE KEY UPDATE
           style of UPDATE, unless values are manually specified here.

        :param \\*args: As an alternative to passing key/value parameters,
         a dictionary or list of 2-tuples can be passed as a single positional
         argument.

         Passing a single dictionary is equivalent to the keyword argument
         form::

            insert().on_duplicate_key_update({"name": "some name"})

         Passing a list of 2-tuples indicates that the parameter assignments
         in the UPDATE clause should be ordered as sent, in a manner similar
         to that described for the :class:`_expression.Update`
         construct overall
         in :ref:`tutorial_parameter_ordered_updates`::

            insert().on_duplicate_key_update(
                [("name", "some name"), ("value", "some value")])

         .. versionchanged:: 1.3 parameters can be specified as a dictionary
            or list of 2-tuples; the latter form provides for parameter
            ordering.


        .. versionadded:: 1.2

        .. seealso::

            :ref:`mysql_insert_on_duplicate_key_update`

        """
    if args and kw:
        raise exc.ArgumentError("Can't pass kwargs and positional arguments simultaneously")
    if args:
        if len(args) > 1:
            raise exc.ArgumentError('Only a single dictionary or list of tuples is accepted positionally.')
        values = args[0]
    else:
        values = kw
    self._post_values_clause = OnDuplicateClause(self.inserted_alias, values)
    return self