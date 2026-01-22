import abc
import dataclasses
import enum
import typing
import warnings
from spnego._credential import Credential
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import FeatureMissingError, NegotiateOptions, SpnegoError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
@staticmethod
def from_oid(oid: str) -> 'GSSMech':
    """Converts an OID string to a GSSMech value.

        Converts an OID string to a GSSMech value if it is known.

        Args:
            oid: The OID as a string to convert from.

        Raises:
            ValueError: if the OID is not a known GSSMech.
        """
    for mech in GSSMech:
        if mech.value == oid:
            return mech
    else:
        raise ValueError("'%s' is not a valid GSSMech OID" % oid)