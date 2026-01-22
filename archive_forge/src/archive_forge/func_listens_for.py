from __future__ import annotations
from typing import Any
from typing import Callable
from .base import _registrars
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
def listens_for(target: Any, identifier: str, *args: Any, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function as a listener for the given target + identifier.

    The :func:`.listens_for` decorator is part of the primary interface for the
    SQLAlchemy event system, documented at :ref:`event_toplevel`.

    This function generally shares the same kwargs as :func:`.listens`.

    e.g.::

        from sqlalchemy import event
        from sqlalchemy.schema import UniqueConstraint

        @event.listens_for(UniqueConstraint, "after_parent_attach")
        def unique_constraint_name(const, table):
            const.name = "uq_%s_%s" % (
                table.name,
                list(const.columns)[0].name
            )

    A given function can also be invoked for only the first invocation
    of the event using the ``once`` argument::

        @event.listens_for(Mapper, "before_configure", once=True)
        def on_config():
            do_config()


    .. warning:: The ``once`` argument does not imply automatic de-registration
       of the listener function after it has been invoked a first time; a
       listener entry will remain associated with the target object.
       Associating an arbitrarily high number of listeners without explicitly
       removing them will cause memory to grow unbounded even if ``once=True``
       is specified.

    .. seealso::

        :func:`.listen` - general description of event listening

    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        listen(target, identifier, fn, *args, **kw)
        return fn
    return decorate