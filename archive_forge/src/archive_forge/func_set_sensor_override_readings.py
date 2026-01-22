from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_sensor_override_readings(type_: SensorType, reading: SensorReading) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Updates the sensor readings reported by a sensor type previously overriden
    by setSensorOverrideEnabled.

    **EXPERIMENTAL**

    :param type_:
    :param reading:
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_.to_json()
    params['reading'] = reading.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setSensorOverrideReadings', 'params': params}
    json = (yield cmd_dict)