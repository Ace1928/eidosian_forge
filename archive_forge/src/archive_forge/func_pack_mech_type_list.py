import enum
import struct
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import KerberosV5Msg
from spnego._ntlm_raw.messages import NTLMMessage
def pack_mech_type_list(mech_list: typing.Union[str, typing.List[str], typing.Tuple[str, ...], typing.Set[str]]) -> bytes:
    """Packs a list of OIDs for the mechListMIC value.

    Will pack a list of object identifiers to the raw byte string value for the mechListMIC.

    Args:
        mech_list: The list of OIDs to back

    Returns:
        bytes: The byte string of the packed ASN.1 MechTypeList SEQUENCE OF value.
    """
    if not isinstance(mech_list, (list, tuple, set)):
        mech_list = [mech_list]
    return pack_asn1_sequence([pack_asn1_object_identifier(oid) for oid in mech_list])