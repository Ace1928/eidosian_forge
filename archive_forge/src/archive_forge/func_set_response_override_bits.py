from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_response_override_bits(authenticator_id: AuthenticatorId, is_bogus_signature: typing.Optional[bool]=None, is_bad_uv: typing.Optional[bool]=None, is_bad_up: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Resets parameters isBogusSignature, isBadUV, isBadUP to false if they are not present.

    :param authenticator_id:
    :param is_bogus_signature: *(Optional)* If isBogusSignature is set, overrides the signature in the authenticator response to be zero. Defaults to false.
    :param is_bad_uv: *(Optional)* If isBadUV is set, overrides the UV bit in the flags in the authenticator response to be zero. Defaults to false.
    :param is_bad_up: *(Optional)* If isBadUP is set, overrides the UP bit in the flags in the authenticator response to be zero. Defaults to false.
    """
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    if is_bogus_signature is not None:
        params['isBogusSignature'] = is_bogus_signature
    if is_bad_uv is not None:
        params['isBadUV'] = is_bad_uv
    if is_bad_up is not None:
        params['isBadUP'] = is_bad_up
    cmd_dict: T_JSON_DICT = {'method': 'WebAuthn.setResponseOverrideBits', 'params': params}
    json = (yield cmd_dict)