import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
import requests
import webob
from heat.api.aws import exception
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
@property
def ssl_options(self):
    if not self._ssl_options:
        cacert = self._conf_get('ca_file')
        insecure = self._conf_get('insecure')
        cert = self._conf_get('cert_file')
        key = self._conf_get('key_file')
        self._ssl_options = {'verify': cacert if cacert else not insecure, 'cert': (cert, key) if cert else None}
    return self._ssl_options