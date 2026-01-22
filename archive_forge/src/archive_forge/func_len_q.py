import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors
def len_q(bstring):
    return sum((len(_q_byte_map[x]) for x in bstring))