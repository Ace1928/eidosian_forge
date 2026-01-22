import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def normalize_inbound_headers(headers, hdr_validation_flags):
    """
    Normalizes a header sequence that we have received.

    :param headers: The HTTP header set.
    :param hdr_validation_flags: An instance of HeaderValidationFlags
    """
    headers = _combine_cookie_fields(headers, hdr_validation_flags)
    return headers