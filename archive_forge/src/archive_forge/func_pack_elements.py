import enum
import struct
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import KerberosV5Msg
from spnego._ntlm_raw.messages import NTLMMessage
def pack_elements(value_map: typing.Iterable[typing.Tuple[int, typing.Any, typing.Callable]]) -> typing.List[bytes]:
    elements = []
    for tag, value, pack_func in value_map:
        if value is not None:
            elements.append(pack_asn1(TagClass.context_specific, True, tag, pack_func(value)))
    return elements