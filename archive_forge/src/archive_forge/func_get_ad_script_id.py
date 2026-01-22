from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_ad_script_id(frame_id: FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[AdScriptId]]:
    """


    **EXPERIMENTAL**

    :param frame_id:
    :returns: *(Optional)* Identifies the bottom-most script which caused the frame to be labelled as an ad. Only sent if frame is labelled as an ad and id is available.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.getAdScriptId', 'params': params}
    json = (yield cmd_dict)
    return AdScriptId.from_json(json['adScriptId']) if 'adScriptId' in json else None