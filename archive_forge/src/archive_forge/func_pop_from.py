import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def pop_from(self, i=0):
    popped = self[i:]
    self[i:] = []
    return popped