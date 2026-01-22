from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def get_histogram(name: str, delta: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, Histogram]:
    """
    Get a Chrome histogram by name.

    **EXPERIMENTAL**

    :param name: Requested histogram name.
    :param delta: *(Optional)* If true, retrieve delta since last delta call.
    :returns: Histogram.
    """
    params: T_JSON_DICT = dict()
    params['name'] = name
    if delta is not None:
        params['delta'] = delta
    cmd_dict: T_JSON_DICT = {'method': 'Browser.getHistogram', 'params': params}
    json = (yield cmd_dict)
    return Histogram.from_json(json['histogram'])