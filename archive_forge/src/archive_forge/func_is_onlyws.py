import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def is_onlyws(self):
    return self._initial_size == 0 and (not self or str(self).isspace())