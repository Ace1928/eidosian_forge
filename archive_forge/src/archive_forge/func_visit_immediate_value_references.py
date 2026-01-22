from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from ..core.has_props import HasProps, Qualified
from ..util.dataclasses import entries, is_dataclass
def visit_immediate_value_references(value: Any, visitor: Callable[[Model], None]) -> None:
    """ Visit all references to another Model without recursing into any
    of the child Model; may visit the same Model more than once if
    it's referenced more than once. Does not visit the passed-in value.

    """
    if isinstance(value, HasProps):
        for attr in value.properties_with_refs():
            child = getattr(value, attr)
            visit_value_and_its_immediate_references(child, visitor)
    else:
        visit_value_and_its_immediate_references(value, visitor)