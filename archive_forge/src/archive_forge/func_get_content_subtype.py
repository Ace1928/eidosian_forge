import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def get_content_subtype(self):
    """Returns the message's sub-content type.

        This is the `subtype' part of the string returned by
        get_content_type().
        """
    ctype = self.get_content_type()
    return ctype.split('/')[1]