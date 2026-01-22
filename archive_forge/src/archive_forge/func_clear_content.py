import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def clear_content(self):
    self._headers = [(n, v) for n, v in self._headers if not n.lower().startswith('content-')]
    self._payload = None