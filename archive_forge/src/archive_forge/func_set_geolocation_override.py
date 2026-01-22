from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_geolocation_override(latitude: typing.Optional[float]=None, longitude: typing.Optional[float]=None, accuracy: typing.Optional[float]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Overrides the Geolocation Position or Error. Omitting any of the parameters emulates position
    unavailable.

    :param latitude: *(Optional)* Mock latitude
    :param longitude: *(Optional)* Mock longitude
    :param accuracy: *(Optional)* Mock accuracy
    """
    params: T_JSON_DICT = dict()
    if latitude is not None:
        params['latitude'] = latitude
    if longitude is not None:
        params['longitude'] = longitude
    if accuracy is not None:
        params['accuracy'] = accuracy
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setGeolocationOverride', 'params': params}
    json = (yield cmd_dict)