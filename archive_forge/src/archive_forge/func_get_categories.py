from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
def get_categories() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    Gets supported tracing categories.

    :returns: A list of supported tracing categories.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Tracing.getCategories'}
    json = (yield cmd_dict)
    return [str(i) for i in json['categories']]