from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def parse_cmd_pkt(line):
    splice_at = line.find(b' ')
    cmd, args = (line[:splice_at], line[splice_at + 1:])
    assert args[-1:] == b'\x00'
    return (cmd, args[:-1].split(b'\x00'))