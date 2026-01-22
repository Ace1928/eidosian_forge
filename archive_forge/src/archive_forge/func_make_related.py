import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def make_related(self, boundary=None):
    self._make_multipart('related', ('alternative', 'mixed'), boundary)