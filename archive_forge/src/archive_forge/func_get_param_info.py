from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
def get_param_info(sig: Signature) -> tuple[list[str], list[Any]]:
    """ Find parameters with defaults and return them.

    Arguments:
        sig (Signature) : a function signature

    Returns:
        tuple(list, list) : parameters with defaults

    """
    defaults = []
    for param in sig.parameters.values():
        if param.default is not param.empty:
            defaults.append(param.default)
    return ([name for name in sig.parameters], defaults)