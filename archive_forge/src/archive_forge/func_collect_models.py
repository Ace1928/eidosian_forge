from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from ..core.has_props import HasProps, Qualified
from ..util.dataclasses import entries, is_dataclass
def collect_models(*input_values: Any) -> list[Model]:
    """ Collect a duplicate-free list of all other Bokeh models referred to by
    this model, or by any of its references, etc.

    Iterate over ``input_values`` and descend through their structure
    collecting all nested ``Models`` on the go. The resulting list is
    duplicate-free based on objects' identifiers.

    Args:
        *input_values (Model)
            Bokeh models to collect other models from

    Returns:
        list[Model] : all models reachable from this one.

    """
    return collect_filtered_models(None, *input_values)