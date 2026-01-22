import http.client
import io
import logging
import math
import urllib.parse
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import identity as ks_identity
from keystoneauth1 import session as ks_session
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store.common import utils as gutils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI
from glance_store import location
@property
def swift_url(self):
    """
        Creates a fully-qualified auth address that the Swift client library
        can use. The scheme for the auth_address is determined using the scheme
        included in the `location` field.

        HTTPS is assumed, unless 'swift+http' is specified.
        """
    if self.auth_or_store_url.startswith('http'):
        return self.auth_or_store_url
    else:
        if self.scheme == 'swift+config':
            if self.ssl_enabled:
                self.scheme = 'swift+https'
            else:
                self.scheme = 'swift+http'
        if self.scheme in ('swift+https', 'swift'):
            auth_scheme = 'https://'
        else:
            auth_scheme = 'http://'
        return ''.join([auth_scheme, self.auth_or_store_url])