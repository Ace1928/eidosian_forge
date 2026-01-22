import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def pkiz_uncompress(signed_text):
    text = signed_text[len(PKIZ_PREFIX):].encode('utf-8')
    unencoded = base64.urlsafe_b64decode(text)
    uncompressed = zlib.decompress(unencoded)
    return uncompressed