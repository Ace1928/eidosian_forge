from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def maybe_in_subresource(segments: Sequence[int | str], resolver: _Resolver[Any], subresource: Resource[Any]) -> _Resolver[Any]:
    _segments = iter(segments)
    for segment in _segments:
        if segment in {'items', 'dependencies'} and isinstance(subresource.contents, Mapping):
            return resolver.in_subresource(subresource)
        if segment not in in_value and (segment not in in_child or next(_segments, None) is None):
            return resolver
    return resolver.in_subresource(subresource)