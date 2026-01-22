import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def make_field_name(parts: Sequence[str]) -> str:
    """Join parts of a field name together. Used for nesting.

    ('parent_1', 'child') => 'parent-1.child'
    ('parents', '1', '_child_node') => 'parents.1._child-node'
    ('parents', '1', 'middle._child_node') => 'parents.1.middle._child-node'
    """
    out: List[str] = []
    for p in _strip_dummy_field_names(parts):
        out.extend(map(replace_delimeter_in_part, p.split('.')))
    return '.'.join(out)