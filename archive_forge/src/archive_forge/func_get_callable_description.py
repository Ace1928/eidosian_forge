import collections.abc
import dataclasses
import functools
import inspect
import io
import itertools
import tokenize
from typing import Callable, Dict, Generic, Hashable, List, Optional, Type, TypeVar
import docstring_parser
from typing_extensions import get_origin, is_typeddict
from . import _resolver, _strings, _unsafe_cache
@_unsafe_cache.unsafe_cache(1024)
def get_callable_description(f: Callable) -> str:
    """Get description associated with a callable via docstring parsing.

    Note that the `dataclasses.dataclass` will automatically populate __doc__ based on
    the fields of the class if a docstring is not specified; this helper will ignore
    these docstrings."""
    f, _unused = _resolver.resolve_generic_types(f)
    f = _resolver.unwrap_origin_strip_extras(f)
    if f in _callable_description_blocklist:
        return ''
    if isinstance(f, functools.partial):
        f = f.func
    try:
        import pydantic
    except ImportError:
        pydantic = None
    docstring = f.__doc__
    if docstring is None and inspect.isclass(f) and (not is_typeddict(f)) and (not _resolver.is_namedtuple(f)) and (not (pydantic is not None and f.__init__ is pydantic.BaseModel.__init__)):
        docstring = f.__init__.__doc__
    if docstring is None:
        return ''
    docstring = _strings.dedent(docstring)
    if dataclasses.is_dataclass(f):
        default_doc = f.__name__ + str(inspect.signature(f)).replace(' -> None', '')
        if docstring == default_doc:
            return ''
    parsed_docstring = docstring_parser.parse(docstring)
    return '\n'.join(list(filter(lambda x: x is not None, [parsed_docstring.short_description, parsed_docstring.long_description])))