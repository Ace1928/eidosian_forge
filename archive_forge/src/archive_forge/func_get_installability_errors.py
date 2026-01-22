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
def get_installability_errors() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[InstallabilityError]]:
    """


    **EXPERIMENTAL**

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getInstallabilityErrors'}
    json = (yield cmd_dict)
    return [InstallabilityError.from_json(i) for i in json['installabilityErrors']]