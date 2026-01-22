import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def normalize_outbound_headers(headers, hdr_validation_flags):
    """
    Normalizes a header sequence that we are about to send.

    :param headers: The HTTP header set.
    :param hdr_validation_flags: An instance of HeaderValidationFlags.
    """
    headers = _lowercase_header_names(headers, hdr_validation_flags)
    headers = _strip_surrounding_whitespace(headers, hdr_validation_flags)
    headers = _strip_connection_headers(headers, hdr_validation_flags)
    headers = _secure_headers(headers, hdr_validation_flags)
    return headers