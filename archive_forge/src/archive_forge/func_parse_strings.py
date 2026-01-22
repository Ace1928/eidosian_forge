from functools import lru_cache
from typing import Callable, Iterable, Iterator, TypeVar, Union, overload
import setuptools.extern.jaraco.text as text
from setuptools.extern.packaging.requirements import Requirement
def parse_strings(strs: _StrOrIter) -> Iterator[str]:
    """
    Yield requirement strings for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return text.join_continuation(map(text.drop_comment, text.yield_lines(strs)))