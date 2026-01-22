import enum
import struct
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import KerberosV5Msg
from spnego._ntlm_raw.messages import NTLMMessage
def unpack_token(b_data: bytes, mech: typing.Optional[GSSMech]=None, unwrap: bool=False, encoding: typing.Optional[str]=None) -> typing.Any:
    """Unpacks a raw GSSAPI/SPNEGO token to a Python object.

    Unpacks the byte string into a Python object that represents the token passed in. This can return many different
    token types such as:

    * NTLM message(s)
    * SPNEGO/Negotiate init or response
    * Kerberos message(s)

    Args:
        b_data: The raw byte string to unpack.
        mech: A hint as to what the byte string is for.
        unwrap: Whether to unwrap raw bytes to a structured message or return the raw tokens bytes.
        encoding: Optional encoding used when unwrapping NTLM messages.

    Returns:
        any: The unpacked SPNEGO, Kerberos, or NTLM token.
    """
    if b_data.startswith(b'NTLMSSP\x00'):
        if unwrap:
            return NTLMMessage.unpack(b_data, encoding=encoding)
        else:
            return b_data
    if mech and mech.is_kerberos_oid:
        raw_data = unpack_asn1(b_data[2:])[0]
    else:
        raw_data = unpack_asn1(b_data)[0]
    if raw_data.tag_class == TagClass.application and mech and mech.is_kerberos_oid:
        return KerberosV5Msg.unpack(unpack_asn1(raw_data.b_data)[0])
    elif raw_data.tag_class == TagClass.application:
        if raw_data.tag_number != 0:
            raise ValueError('Expecting a tag number of 0 not %s for InitialContextToken' % raw_data.tag_number)
        initial_context_token = InitialContextToken.unpack(raw_data.b_data)
        if unwrap:
            return initial_context_token
        this_mech: typing.Optional[GSSMech]
        try:
            this_mech = GSSMech.from_oid(initial_context_token.this_mech)
        except ValueError:
            this_mech = None
        if this_mech and (this_mech == GSSMech.spnego or (this_mech.is_kerberos_oid and unwrap)):
            return unpack_token(initial_context_token.inner_context_token, mech=this_mech)
        return b_data
    elif raw_data.tag_class == TagClass.context_specific:
        if raw_data.tag_number == 0:
            return NegTokenInit.unpack(raw_data.b_data)
        elif raw_data.tag_number == 1:
            return NegTokenResp.unpack(raw_data.b_data)
        else:
            raise ValueError('Unknown NegotiationToken CHOICE %d, only expecting 0 or 1' % raw_data.tag_number)
    elif unwrap:
        return KerberosV5Msg.unpack(raw_data)
    else:
        return b_data