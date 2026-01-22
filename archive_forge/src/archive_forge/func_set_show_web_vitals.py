from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_web_vitals(show: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Request that backend shows an overlay with web vital metrics.

    :param show:
    """
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowWebVitals', 'params': params}
    json = (yield cmd_dict)