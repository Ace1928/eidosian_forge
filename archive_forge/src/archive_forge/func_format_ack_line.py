from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def format_ack_line(sha, ack_type=b''):
    if ack_type:
        ack_type = b' ' + ack_type
    return b'ACK ' + sha + ack_type + b'\n'