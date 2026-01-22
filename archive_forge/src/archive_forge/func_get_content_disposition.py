import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def get_content_disposition(self):
    """Return the message's content-disposition if it exists, or None.

        The return values can be either 'inline', 'attachment' or None
        according to the rfc2183.
        """
    value = self.get('content-disposition')
    if value is None:
        return None
    c_d = _splitparam(value)[0].lower()
    return c_d