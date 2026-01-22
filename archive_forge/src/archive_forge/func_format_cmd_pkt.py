from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def format_cmd_pkt(cmd, *args):
    return cmd + b' ' + b''.join([a + b'\x00' for a in args])