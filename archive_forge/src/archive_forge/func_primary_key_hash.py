import hashlib
from cryptography import fernet
from oslo_log import log
from keystone.common import fernet_utils
import keystone.conf
from keystone.credential.providers import core
from keystone import exception
from keystone.i18n import _
def primary_key_hash(keys):
    """Calculate a hash of the primary key used for encryption."""
    if isinstance(keys[0], str):
        keys[0] = keys[0].encode('utf-8')
    return hashlib.sha1(keys[0]).hexdigest()