import functools
import sys
import typing as t
from collections import abc
from itertools import chain
from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context
def make_logging_undefined(logger: t.Optional['logging.Logger']=None, base: t.Type[Undefined]=Undefined) -> t.Type[Undefined]:
    """Given a logger object this returns a new undefined class that will
    log certain failures.  It will log iterations and printing.  If no
    logger is given a default logger is created.

    Example::

        logger = logging.getLogger(__name__)
        LoggingUndefined = make_logging_undefined(
            logger=logger,
            base=Undefined
        )

    .. versionadded:: 2.8

    :param logger: the logger to use.  If not provided, a default logger
                   is created.
    :param base: the base class to add logging functionality to.  This
                 defaults to :class:`Undefined`.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler(sys.stderr))

    def _log_message(undef: Undefined) -> None:
        logger.warning('Template variable warning: %s', undef._undefined_message)

    class LoggingUndefined(base):
        __slots__ = ()

        def _fail_with_undefined_error(self, *args: t.Any, **kwargs: t.Any) -> 'te.NoReturn':
            try:
                super()._fail_with_undefined_error(*args, **kwargs)
            except self._undefined_exception as e:
                logger.error('Template variable error: %s', e)
                raise e

        def __str__(self) -> str:
            _log_message(self)
            return super().__str__()

        def __iter__(self) -> t.Iterator[t.Any]:
            _log_message(self)
            return super().__iter__()

        def __bool__(self) -> bool:
            _log_message(self)
            return super().__bool__()
    return LoggingUndefined