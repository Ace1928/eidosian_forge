import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def biluo_to_iob(tags: Iterable[str]) -> List[str]:
    out = []
    for tag in tags:
        if tag is None:
            out.append(tag)
        else:
            tag = tag.replace('U-', 'B-', 1).replace('L-', 'I-', 1)
            out.append(tag)
    return out