import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def get_stores_from_request(req, body):
    """Processes a supplied request and extract stores from it

    :param req: request to process
    :param body: request body

    :raises glance_store.UnknownScheme:  if a store is not valid
    :return: a list of stores
    """
    if body.get('all_stores', False):
        if 'stores' in body or 'x-image-meta-store' in req.headers:
            msg = _("All_stores parameter can't be used with x-image-meta-store header or stores parameter")
            raise exc.HTTPBadRequest(explanation=msg)
        stores = _get_available_stores()
    else:
        try:
            stores = body['stores']
        except KeyError:
            stores = [req.headers.get('x-image-meta-store', CONF.glance_store.default_backend)]
        else:
            if 'x-image-meta-store' in req.headers:
                msg = _("Stores parameter and x-image-meta-store header can't be both specified")
                raise exc.HTTPBadRequest(explanation=msg)
    for store in stores:
        glance_store.get_store_from_store_identifier(store)
    return stores