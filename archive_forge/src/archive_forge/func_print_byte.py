import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def print_byte(byte, pos=0):
    char = chr(byte)
    if char.isspace() or char not in string.printable:
        char = '.'
    print(f'  {pos:06x}: {byte:02x}  {char}')