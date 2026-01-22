import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def validate_outbound_headers(headers, hdr_validation_flags):
    """
    Validates and normalizes a header sequence that we are about to send.

    :param headers: The HTTP header set.
    :param hdr_validation_flags: An instance of HeaderValidationFlags.
    """
    headers = _reject_te(headers, hdr_validation_flags)
    headers = _reject_connection_header(headers, hdr_validation_flags)
    headers = _reject_pseudo_header_fields(headers, hdr_validation_flags)
    headers = _check_sent_host_authority_header(headers, hdr_validation_flags)
    headers = _check_path_header(headers, hdr_validation_flags)
    return headers