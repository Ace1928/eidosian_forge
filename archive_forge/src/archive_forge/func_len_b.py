import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors
def len_b(bstring):
    groups_of_3, leftover = divmod(len(bstring), 3)
    return groups_of_3 * 4 + (4 if leftover else 0)