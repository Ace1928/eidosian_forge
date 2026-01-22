from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
def collect_known_metadata(annotations: Iterable[Any]) -> tuple[dict[str, Any], list[Any]]:
    """Split `annotations` into known metadata and unknown annotations.

    Args:
        annotations: An iterable of annotations.

    Returns:
        A tuple contains a dict of known metadata and a list of unknown annotations.

    Example:
        ```py
        from annotated_types import Gt, Len

        from pydantic._internal._known_annotated_metadata import collect_known_metadata

        print(collect_known_metadata([Gt(1), Len(42), ...]))
        #> ({'gt': 1, 'min_length': 42}, [Ellipsis])
        ```
    """
    import annotated_types as at
    annotations = expand_grouped_metadata(annotations)
    res: dict[str, Any] = {}
    remaining: list[Any] = []
    for annotation in annotations:
        if isinstance(annotation, PydanticMetadata):
            res.update(annotation.__dict__)
        elif isinstance(annotation, at.MinLen):
            res.update({'min_length': annotation.min_length})
        elif isinstance(annotation, at.MaxLen):
            res.update({'max_length': annotation.max_length})
        elif isinstance(annotation, at.Gt):
            res.update({'gt': annotation.gt})
        elif isinstance(annotation, at.Ge):
            res.update({'ge': annotation.ge})
        elif isinstance(annotation, at.Lt):
            res.update({'lt': annotation.lt})
        elif isinstance(annotation, at.Le):
            res.update({'le': annotation.le})
        elif isinstance(annotation, at.MultipleOf):
            res.update({'multiple_of': annotation.multiple_of})
        elif isinstance(annotation, type) and issubclass(annotation, PydanticMetadata):
            res.update({k: v for k, v in vars(annotation).items() if not k.startswith('_')})
        else:
            remaining.append(annotation)
    res = {k: v for k, v in res.items() if v is not None}
    return (res, remaining)