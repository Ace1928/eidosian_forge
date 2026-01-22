import base64
import datetime
import uuid
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.models import token_model
from keystone import notifications
def random_urlsafe_str():
    """Generate a random URL-safe string.

    :rtype: str
    """
    return base64.urlsafe_b64encode(uuid.uuid4().bytes)[:-2].decode('utf-8')