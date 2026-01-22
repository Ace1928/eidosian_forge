from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def start_sampling(sampling_interval: typing.Optional[float]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param sampling_interval: *(Optional)* Average sample interval in bytes. Poisson distribution is used for the intervals. The default value is 32768 bytes.
    """
    params: T_JSON_DICT = dict()
    if sampling_interval is not None:
        params['samplingInterval'] = sampling_interval
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.startSampling', 'params': params}
    json = (yield cmd_dict)