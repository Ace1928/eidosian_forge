from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def pkt_line(data):
    """Wrap data in a pkt-line.

    Args:
      data: The data to wrap, as a str or None.
    Returns: The data prefixed with its length in pkt-line format; if data was
        None, returns the flush-pkt ('0000').
    """
    if data is None:
        return b'0000'
    return ('%04x' % (len(data) + 4)).encode('ascii') + data