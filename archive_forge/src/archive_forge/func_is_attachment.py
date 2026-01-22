import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def is_attachment(self):
    c_d = self.get('content-disposition')
    return False if c_d is None else c_d.content_disposition == 'attachment'