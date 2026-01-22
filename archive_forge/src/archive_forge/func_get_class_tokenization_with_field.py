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
def get_class_tokenization_with_field(cls: Type, field_name: str) -> Optional[_ClassTokenization]:
    found_field: bool = False
    classes_to_search = cls.__mro__
    tokenization = None
    for search_cls in classes_to_search:
        assert search_cls is Generic or get_origin(search_cls) is None
        try:
            tokenization = _ClassTokenization.make(search_cls)
        except OSError as e:
            assert 'could not find class definition' in e.args[0] or 'source code not available' in e.args[0]
            return None
        except TypeError as e:
            assert 'built-in class' in e.args[0]
            return None
        if field_name in tokenization.field_data_from_name:
            found_field = True
            break
    if dataclasses.is_dataclass(cls):
        assert found_field, 'Docstring parsing error -- this usually means that there are multiple dataclasses in the same file with the same name but different scopes.'
    return tokenization