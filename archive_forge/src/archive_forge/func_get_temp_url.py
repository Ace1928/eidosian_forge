import datetime
import email.utils
import hashlib
import logging
import random
import time
from urllib import parse
from oslo_config import cfg
from swiftclient import client as sc
from swiftclient import exceptions
from swiftclient import utils as swiftclient_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
def get_temp_url(self, container_name, obj_name, timeout=None, method='PUT'):
    """Return a Swift TempURL."""
    key_header = 'x-account-meta-temp-url-key'
    if key_header not in self.client().head_account():
        self.client().post_account({key_header: hashlib.sha224(str(random.getrandbits(256)).encode('latin-1')).hexdigest()[:32]})
    key = self.client().head_account()[key_header]
    path = '/v1/AUTH_%s/%s/%s' % (self.context.tenant_id, container_name, obj_name)
    if timeout is None:
        timeout = int(MAX_EPOCH - 60 - time.time())
    tempurl = swiftclient_utils.generate_temp_url(path, timeout, key, method)
    sw_url = parse.urlparse(self.client().url)
    return '%s://%s%s' % (sw_url.scheme, sw_url.netloc, tempurl)