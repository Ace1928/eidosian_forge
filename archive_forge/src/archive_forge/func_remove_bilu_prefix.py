import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def remove_bilu_prefix(label: str) -> str:
    return label.split('-', 1)[1]