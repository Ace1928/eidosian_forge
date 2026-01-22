from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_time_domain(time_domain: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets time domain to use for collecting and reporting duration metrics.
    Note that this must be called before enabling metrics collection. Calling
    this method while metrics collection is enabled returns an error.

    **EXPERIMENTAL**

    :param time_domain: Time domain
    """
    params: T_JSON_DICT = dict()
    params['timeDomain'] = time_domain
    cmd_dict: T_JSON_DICT = {'method': 'Performance.setTimeDomain', 'params': params}
    json = (yield cmd_dict)