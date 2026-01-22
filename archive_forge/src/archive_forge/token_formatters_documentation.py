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
Convert a value to text type, translating uuid -> hex if required.

        :param is_stored_as_bytes: whether value is already bytes
        :type is_stored_as_bytes: boolean
        :param value: value to attempt to convert to bytes
        :type value: str or bytes
        :rtype: str
        