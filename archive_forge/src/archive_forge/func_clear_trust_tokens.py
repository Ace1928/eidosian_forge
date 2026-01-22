from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def clear_trust_tokens(issuer_origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, bool]:
    """
    Removes all Trust Tokens issued by the provided issuerOrigin.
    Leaves other stored data, including the issuer's Redemption Records, intact.

    **EXPERIMENTAL**

    :param issuer_origin:
    :returns: True if any tokens were deleted, false otherwise.
    """
    params: T_JSON_DICT = dict()
    params['issuerOrigin'] = issuer_origin
    cmd_dict: T_JSON_DICT = {'method': 'Storage.clearTrustTokens', 'params': params}
    json = (yield cmd_dict)
    return bool(json['didDeleteTokens'])