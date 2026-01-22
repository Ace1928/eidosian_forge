from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def get_overridden_sensor_information(type_: SensorType) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, float]:
    """


    **EXPERIMENTAL**

    :param type_:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.getOverriddenSensorInformation', 'params': params}
    json = (yield cmd_dict)
    return float(json['requestedSamplingFrequency'])