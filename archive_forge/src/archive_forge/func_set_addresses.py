from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_addresses(addresses: typing.List[Address]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set addresses so that developers can verify their forms implementation.

    :param addresses:
    """
    params: T_JSON_DICT = dict()
    params['addresses'] = [i.to_json() for i in addresses]
    cmd_dict: T_JSON_DICT = {'method': 'Autofill.setAddresses', 'params': params}
    json = (yield cmd_dict)