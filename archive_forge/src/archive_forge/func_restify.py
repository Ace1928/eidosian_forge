import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def restify(cls: Optional[Type], mode: str='fully-qualified-except-typing') -> str:
    """Convert python class to a reST reference.

    :param mode: Specify a method how annotations will be stringified.

                 'fully-qualified-except-typing'
                     Show the module name and qualified name of the annotation except
                     the "typing" module.
                 'smart'
                     Show the name of the annotation.
    """
    from sphinx.ext.autodoc.mock import ismock, ismockmodule
    from sphinx.util import inspect
    if mode == 'smart':
        modprefix = '~'
    else:
        modprefix = ''
    try:
        if cls is None or cls is NoneType:
            return ':py:obj:`None`'
        elif cls is Ellipsis:
            return '...'
        elif isinstance(cls, str):
            return cls
        elif ismockmodule(cls):
            return ':py:class:`%s%s`' % (modprefix, cls.__name__)
        elif ismock(cls):
            return ':py:class:`%s%s.%s`' % (modprefix, cls.__module__, cls.__name__)
        elif is_invalid_builtin_class(cls):
            return ':py:class:`%s%s`' % (modprefix, INVALID_BUILTIN_CLASSES[cls])
        elif inspect.isNewType(cls):
            if sys.version_info > (3, 10):
                return ':py:class:`%s%s.%s`' % (modprefix, cls.__module__, cls.__name__)
            else:
                return ':py:class:`%s`' % cls.__name__
        elif UnionType and isinstance(cls, UnionType):
            if len(cls.__args__) > 1 and None in cls.__args__:
                args = ' | '.join((restify(a, mode) for a in cls.__args__ if a))
                return 'Optional[%s]' % args
            else:
                return ' | '.join((restify(a, mode) for a in cls.__args__))
        elif cls.__module__ in ('__builtin__', 'builtins'):
            if hasattr(cls, '__args__'):
                return ':py:class:`%s`\\ [%s]' % (cls.__name__, ', '.join((restify(arg, mode) for arg in cls.__args__)))
            else:
                return ':py:class:`%s`' % cls.__name__
        elif sys.version_info >= (3, 7):
            return _restify_py37(cls, mode)
        else:
            return _restify_py36(cls, mode)
    except (AttributeError, TypeError):
        return inspect.object_description(cls)