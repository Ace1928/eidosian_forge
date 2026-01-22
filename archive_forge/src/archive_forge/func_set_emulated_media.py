from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_emulated_media(media: typing.Optional[str]=None, features: typing.Optional[typing.List[MediaFeature]]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Emulates the given media type or media feature for CSS media queries.

    :param media: *(Optional)* Media type to emulate. Empty string disables the override.
    :param features: *(Optional)* Media features to emulate.
    """
    params: T_JSON_DICT = dict()
    if media is not None:
        params['media'] = media
    if features is not None:
        params['features'] = [i.to_json() for i in features]
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setEmulatedMedia', 'params': params}
    json = (yield cmd_dict)