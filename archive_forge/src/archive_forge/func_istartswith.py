from __future__ import annotations
from enum import IntEnum
from operator import add as _uncast_add
from operator import and_ as _uncast_and_
from operator import contains as _uncast_contains
from operator import eq as _uncast_eq
from operator import floordiv as _uncast_floordiv
from operator import ge as _uncast_ge
from operator import getitem as _uncast_getitem
from operator import gt as _uncast_gt
from operator import inv as _uncast_inv
from operator import le as _uncast_le
from operator import lshift as _uncast_lshift
from operator import lt as _uncast_lt
from operator import mod as _uncast_mod
from operator import mul as _uncast_mul
from operator import ne as _uncast_ne
from operator import neg as _uncast_neg
from operator import or_ as _uncast_or_
from operator import rshift as _uncast_rshift
from operator import sub as _uncast_sub
from operator import truediv as _uncast_truediv
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def istartswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnOperators:
    """Implement the ``istartswith`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.startswith`.

        Produces a LIKE expression that tests against an insensitive
        match for the start of a string value::

            lower(column) LIKE lower(<other>) || '%'

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.istartswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.istartswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.istartswith.escape` parameter will
        establish a given character as an escape character which can be of
        use when the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.istartswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.istartswith("foo%bar", autoescape=True)

          Will render as::

            lower(somecolumn) LIKE lower(:param) || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.istartswith("foo/%bar", escape="^")

          Will render as::

            lower(somecolumn) LIKE lower(:param) || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.istartswith.autoescape`::

            somecolumn.istartswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`
        """
    return self.operate(istartswith_op, other, escape=escape, autoescape=autoescape)