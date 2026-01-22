from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def namespace_candidate(namespace: str, matcher: Callable[..., bool]) -> Callable[..., bool]:

    def _matcher(obj: Any, *args: Any, **kwargs: Any) -> bool:
        if is_namespace_extension(obj) and obj[0] == namespace:
            return matcher(obj[1], *args, **kwargs)
        return False
    return _matcher