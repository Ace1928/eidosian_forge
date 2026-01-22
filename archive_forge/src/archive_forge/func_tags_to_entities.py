import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def tags_to_entities(tags: Iterable[str]) -> List[Tuple[str, int, int]]:
    """Note that the end index returned by this function is inclusive.
    To use it for Span creation, increment the end by 1."""
    entities = []
    start = None
    for i, tag in enumerate(tags):
        if tag is None or tag.startswith('-'):
            if start is not None:
                start = None
            else:
                entities.append(('', i, i))
        elif tag.startswith('O'):
            pass
        elif tag.startswith('I'):
            if start is None:
                raise ValueError(Errors.E067.format(start='I', tags=list(tags)[:i + 1]))
        elif tag.startswith('U'):
            entities.append((tag[2:], i, i))
        elif tag.startswith('B'):
            start = i
        elif tag.startswith('L'):
            if start is None:
                raise ValueError(Errors.E067.format(start='L', tags=list(tags)[:i + 1]))
            entities.append((tag[2:], start, i))
            start = None
        else:
            raise ValueError(Errors.E068.format(tag=tag))
    return entities