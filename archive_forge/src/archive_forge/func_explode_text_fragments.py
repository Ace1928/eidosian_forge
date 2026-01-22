from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, List, TypeVar, cast, overload
from prompt_toolkit.formatted_text.base import OneStyleAndTextTuple
def explode_text_fragments(fragments: Iterable[_T]) -> _ExplodedList[_T]:
    """
    Turn a list of (style_str, text) tuples into another list where each string is
    exactly one character.

    It should be fine to call this function several times. Calling this on a
    list that is already exploded, is a null operation.

    :param fragments: List of (style, text) tuples.
    """
    if isinstance(fragments, _ExplodedList):
        return fragments
    result: list[_T] = []
    for style, string, *rest in fragments:
        for c in string:
            result.append((style, c, *rest))
    return _ExplodedList(result)