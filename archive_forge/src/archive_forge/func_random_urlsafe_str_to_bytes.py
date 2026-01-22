import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def random_urlsafe_str_to_bytes(cls, s):
    """Convert string from :func:`random_urlsafe_str()` to bytes.

        :type s: str
        :rtype: bytes

        """
    s = str(s)
    return base64.urlsafe_b64decode(s + '==')