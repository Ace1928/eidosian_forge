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
def swift_retry_iter(resp_iter, length, store, location, manager):
    if not length and isinstance(resp_iter, io.BytesIO):
        pos = resp_iter.tell()
        resp_iter.seek(0, 2)
        length = resp_iter.tell()
        resp_iter.seek(pos)
    length = length if length else resp_iter.len if hasattr(resp_iter, 'len') else 0
    retries = 0
    bytes_read = 0
    if store.backend_group:
        rcount = getattr(store.conf, store.backend_group).swift_store_retry_get_count
    else:
        rcount = store.conf.glance_store.swift_store_retry_get_count
    while retries <= rcount:
        try:
            for chunk in resp_iter:
                yield chunk
                bytes_read += len(chunk)
        except swiftclient.ClientException as e:
            LOG.warning('Swift exception raised %s' % encodeutils.exception_to_unicode(e))
        if bytes_read != length:
            if retries == rcount:
                LOG.error(_LE('Stopping Swift retries after %d attempts') % retries)
                break
            else:
                retries += 1
                LOG.info(_LI('Retrying Swift connection (%(retries)d/%(max_retries)d) with range=%(start)d-%(end)d'), {'retries': retries, 'max_retries': rcount, 'start': bytes_read, 'end': length})
                _resp_headers, resp_iter = store._get_object(location, manager, bytes_read)
        else:
            break