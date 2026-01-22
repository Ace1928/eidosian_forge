from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def format_unshallow_line(sha):
    return COMMAND_UNSHALLOW + b' ' + sha