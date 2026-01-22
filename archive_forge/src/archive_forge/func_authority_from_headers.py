import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def authority_from_headers(headers):
    """
    Given a header set, searches for the authority header and returns the
    value.

    Note that this doesn't terminate early, so should only be called if the
    headers are for a client request. Otherwise, will loop over the entire
    header set, which is potentially unwise.

    :param headers: The HTTP header set.
    :returns: The value of the authority header, or ``None``.
    :rtype: ``bytes`` or ``None``.
    """
    for n, v in headers:
        if n in (b':authority', u':authority'):
            return v.encode('utf-8') if not isinstance(v, bytes) else v
    return None