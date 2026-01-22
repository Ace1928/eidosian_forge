from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
def set_override_certificate_errors(override: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enable/disable overriding certificate errors. If enabled, all certificate error events need to
    be handled by the DevTools client and should be answered with ``handleCertificateError`` commands.

    :param override: If true, certificate errors will be overridden.
    """
    params: T_JSON_DICT = dict()
    params['override'] = override
    cmd_dict: T_JSON_DICT = {'method': 'Security.setOverrideCertificateErrors', 'params': params}
    json = (yield cmd_dict)