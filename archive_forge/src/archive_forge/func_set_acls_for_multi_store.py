import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def set_acls_for_multi_store(location_uri, backend, public=False, read_tenants=[], write_tenants=None, context=None):
    if write_tenants is None:
        write_tenants = []
    loc = location.get_location_from_uri_and_backend(location_uri, backend, conf=CONF)
    store = get_store_from_store_identifier(backend)
    try:
        store.set_acls(loc, public=public, read_tenants=read_tenants, write_tenants=write_tenants, context=context)
    except NotImplementedError:
        LOG.debug('Skipping store.set_acls... not implemented')