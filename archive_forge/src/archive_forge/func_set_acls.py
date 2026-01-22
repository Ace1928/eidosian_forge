from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
def set_acls(self, uri, public=False, read_tenants=None, write_tenants=None, context=None):
    if read_tenants is None:
        read_tenants = []
    if write_tenants is None:
        write_tenants = []
    self.acls[uri] = {'public': public, 'read': read_tenants, 'write': write_tenants}