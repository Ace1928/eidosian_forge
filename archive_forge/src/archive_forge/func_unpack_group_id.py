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
def unpack_group_id(cls, group_id_in_bytes):
    is_stored_as_bytes, group_id = group_id_in_bytes
    group_id = cls._convert_or_decode(is_stored_as_bytes, group_id)
    return {'id': group_id}