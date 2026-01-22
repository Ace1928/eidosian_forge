from base64 import b64encode, b64decode
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_unicode
import passlib.utils.handlers as uh
from passlib.utils.compat import bascii_to_str, iteritems, u,\
from passlib.crypto.digest import pbkdf1
@property
def checksum_size(self):
    return self._variant_info[self.variant][1]