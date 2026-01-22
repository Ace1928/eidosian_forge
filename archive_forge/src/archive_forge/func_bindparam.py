from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple as typing_Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from .base import _NoArg
from .coercions import _document_text_coercion
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import Case
from .elements import Cast
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Extract
from .elements import False_
from .elements import FunctionFilter
from .elements import Label
from .elements import Null
from .elements import Over
from .elements import TextClause
from .elements import True_
from .elements import TryCast
from .elements import Tuple
from .elements import TypeCoerce
from .elements import UnaryExpression
from .elements import WithinGroup
from .functions import FunctionElement
from ..util.typing import Literal
def bindparam(key: Optional[str], value: Any=_NoArg.NO_ARG, type_: Optional[_TypeEngineArgument[_T]]=None, unique: bool=False, required: Union[bool, Literal[_NoArg.NO_ARG]]=_NoArg.NO_ARG, quote: Optional[bool]=None, callable_: Optional[Callable[[], Any]]=None, expanding: bool=False, isoutparam: bool=False, literal_execute: bool=False) -> BindParameter[_T]:
    """Produce a "bound expression".

    The return value is an instance of :class:`.BindParameter`; this
    is a :class:`_expression.ColumnElement`
    subclass which represents a so-called
    "placeholder" value in a SQL expression, the value of which is
    supplied at the point at which the statement in executed against a
    database connection.

    In SQLAlchemy, the :func:`.bindparam` construct has
    the ability to carry along the actual value that will be ultimately
    used at expression time.  In this way, it serves not just as
    a "placeholder" for eventual population, but also as a means of
    representing so-called "unsafe" values which should not be rendered
    directly in a SQL statement, but rather should be passed along
    to the :term:`DBAPI` as values which need to be correctly escaped
    and potentially handled for type-safety.

    When using :func:`.bindparam` explicitly, the use case is typically
    one of traditional deferment of parameters; the :func:`.bindparam`
    construct accepts a name which can then be referred to at execution
    time::

        from sqlalchemy import bindparam

        stmt = select(users_table).where(
            users_table.c.name == bindparam("username")
        )

    The above statement, when rendered, will produce SQL similar to::

        SELECT id, name FROM user WHERE name = :username

    In order to populate the value of ``:username`` above, the value
    would typically be applied at execution time to a method
    like :meth:`_engine.Connection.execute`::

        result = connection.execute(stmt, {"username": "wendy"})

    Explicit use of :func:`.bindparam` is also common when producing
    UPDATE or DELETE statements that are to be invoked multiple times,
    where the WHERE criterion of the statement is to change on each
    invocation, such as::

        stmt = (
            users_table.update()
            .where(user_table.c.name == bindparam("username"))
            .values(fullname=bindparam("fullname"))
        )

        connection.execute(
            stmt,
            [
                {"username": "wendy", "fullname": "Wendy Smith"},
                {"username": "jack", "fullname": "Jack Jones"},
            ],
        )

    SQLAlchemy's Core expression system makes wide use of
    :func:`.bindparam` in an implicit sense.   It is typical that Python
    literal values passed to virtually all SQL expression functions are
    coerced into fixed :func:`.bindparam` constructs.  For example, given
    a comparison operation such as::

        expr = users_table.c.name == 'Wendy'

    The above expression will produce a :class:`.BinaryExpression`
    construct, where the left side is the :class:`_schema.Column` object
    representing the ``name`` column, and the right side is a
    :class:`.BindParameter` representing the literal value::

        print(repr(expr.right))
        BindParameter('%(4327771088 name)s', 'Wendy', type_=String())

    The expression above will render SQL such as::

        user.name = :name_1

    Where the ``:name_1`` parameter name is an anonymous name.  The
    actual string ``Wendy`` is not in the rendered string, but is carried
    along where it is later used within statement execution.  If we
    invoke a statement like the following::

        stmt = select(users_table).where(users_table.c.name == 'Wendy')
        result = connection.execute(stmt)

    We would see SQL logging output as::

        SELECT "user".id, "user".name
        FROM "user"
        WHERE "user".name = %(name_1)s
        {'name_1': 'Wendy'}

    Above, we see that ``Wendy`` is passed as a parameter to the database,
    while the placeholder ``:name_1`` is rendered in the appropriate form
    for the target database, in this case the PostgreSQL database.

    Similarly, :func:`.bindparam` is invoked automatically when working
    with :term:`CRUD` statements as far as the "VALUES" portion is
    concerned.   The :func:`_expression.insert` construct produces an
    ``INSERT`` expression which will, at statement execution time, generate
    bound placeholders based on the arguments passed, as in::

        stmt = users_table.insert()
        result = connection.execute(stmt, {"name": "Wendy"})

    The above will produce SQL output as::

        INSERT INTO "user" (name) VALUES (%(name)s)
        {'name': 'Wendy'}

    The :class:`_expression.Insert` construct, at
    compilation/execution time, rendered a single :func:`.bindparam`
    mirroring the column name ``name`` as a result of the single ``name``
    parameter we passed to the :meth:`_engine.Connection.execute` method.

    :param key:
      the key (e.g. the name) for this bind param.
      Will be used in the generated
      SQL statement for dialects that use named parameters.  This
      value may be modified when part of a compilation operation,
      if other :class:`BindParameter` objects exist with the same
      key, or if its length is too long and truncation is
      required.

      If omitted, an "anonymous" name is generated for the bound parameter;
      when given a value to bind, the end result is equivalent to calling upon
      the :func:`.literal` function with a value to bind, particularly
      if the :paramref:`.bindparam.unique` parameter is also provided.

    :param value:
      Initial value for this bind param.  Will be used at statement
      execution time as the value for this parameter passed to the
      DBAPI, if no other value is indicated to the statement execution
      method for this particular parameter name.  Defaults to ``None``.

    :param callable\\_:
      A callable function that takes the place of "value".  The function
      will be called at statement execution time to determine the
      ultimate value.   Used for scenarios where the actual bind
      value cannot be determined at the point at which the clause
      construct is created, but embedded bind values are still desirable.

    :param type\\_:
      A :class:`.TypeEngine` class or instance representing an optional
      datatype for this :func:`.bindparam`.  If not passed, a type
      may be determined automatically for the bind, based on the given
      value; for example, trivial Python types such as ``str``,
      ``int``, ``bool``
      may result in the :class:`.String`, :class:`.Integer` or
      :class:`.Boolean` types being automatically selected.

      The type of a :func:`.bindparam` is significant especially in that
      the type will apply pre-processing to the value before it is
      passed to the database.  For example, a :func:`.bindparam` which
      refers to a datetime value, and is specified as holding the
      :class:`.DateTime` type, may apply conversion needed to the
      value (such as stringification on SQLite) before passing the value
      to the database.

    :param unique:
      if True, the key name of this :class:`.BindParameter` will be
      modified if another :class:`.BindParameter` of the same name
      already has been located within the containing
      expression.  This flag is used generally by the internals
      when producing so-called "anonymous" bound expressions, it
      isn't generally applicable to explicitly-named :func:`.bindparam`
      constructs.

    :param required:
      If ``True``, a value is required at execution time.  If not passed,
      it defaults to ``True`` if neither :paramref:`.bindparam.value`
      or :paramref:`.bindparam.callable` were passed.  If either of these
      parameters are present, then :paramref:`.bindparam.required`
      defaults to ``False``.

    :param quote:
      True if this parameter name requires quoting and is not
      currently known as a SQLAlchemy reserved word; this currently
      only applies to the Oracle backend, where bound names must
      sometimes be quoted.

    :param isoutparam:
      if True, the parameter should be treated like a stored procedure
      "OUT" parameter.  This applies to backends such as Oracle which
      support OUT parameters.

    :param expanding:
      if True, this parameter will be treated as an "expanding" parameter
      at execution time; the parameter value is expected to be a sequence,
      rather than a scalar value, and the string SQL statement will
      be transformed on a per-execution basis to accommodate the sequence
      with a variable number of parameter slots passed to the DBAPI.
      This is to allow statement caching to be used in conjunction with
      an IN clause.

      .. seealso::

        :meth:`.ColumnOperators.in_`

        :ref:`baked_in` - with baked queries

      .. note:: The "expanding" feature does not support "executemany"-
         style parameter sets.

      .. versionadded:: 1.2

      .. versionchanged:: 1.3 the "expanding" bound parameter feature now
         supports empty lists.

    :param literal_execute:
      if True, the bound parameter will be rendered in the compile phase
      with a special "POSTCOMPILE" token, and the SQLAlchemy compiler will
      render the final value of the parameter into the SQL statement at
      statement execution time, omitting the value from the parameter
      dictionary / list passed to DBAPI ``cursor.execute()``.  This
      produces a similar effect as that of using the ``literal_binds``,
      compilation flag,  however takes place as the statement is sent to
      the DBAPI ``cursor.execute()`` method, rather than when the statement
      is compiled.   The primary use of this
      capability is for rendering LIMIT / OFFSET clauses for database
      drivers that can't accommodate for bound parameters in these
      contexts, while allowing SQL constructs to be cacheable at the
      compilation level.

      .. versionadded:: 1.4 Added "post compile" bound parameters

        .. seealso::

            :ref:`change_4808`.

    .. seealso::

        :ref:`tutorial_sending_parameters` - in the
        :ref:`unified_tutorial`


    """
    return BindParameter(key, value, type_, unique, required, quote, callable_, expanding, isoutparam, literal_execute)