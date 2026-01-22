from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def get_histograms(query: typing.Optional[str]=None, delta: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[Histogram]]:
    """
    Get Chrome histograms.

    **EXPERIMENTAL**

    :param query: *(Optional)* Requested substring in name. Only histograms which have query as a substring in their name are extracted. An empty or absent query returns all histograms.
    :param delta: *(Optional)* If true, retrieve delta since last delta call.
    :returns: Histograms.
    """
    params: T_JSON_DICT = dict()
    if query is not None:
        params['query'] = query
    if delta is not None:
        params['delta'] = delta
    cmd_dict: T_JSON_DICT = {'method': 'Browser.getHistograms', 'params': params}
    json = (yield cmd_dict)
    return [Histogram.from_json(i) for i in json['histograms']]