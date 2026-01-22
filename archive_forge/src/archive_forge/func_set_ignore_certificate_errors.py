from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
def set_ignore_certificate_errors(ignore: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enable/disable whether all certificate errors should be ignored.

    :param ignore: If true, all certificate errors will be ignored.
    """
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {'method': 'Security.setIgnoreCertificateErrors', 'params': params}
    json = (yield cmd_dict)