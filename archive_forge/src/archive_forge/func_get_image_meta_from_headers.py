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
def get_image_meta_from_headers(response):
    """
    Processes HTTP headers from a supplied response that
    match the x-image-meta and x-image-meta-property and
    returns a mapping of image metadata and properties

    :param response: Response to process
    """
    result = {}
    properties = {}
    if hasattr(response, 'getheaders'):
        headers = response.getheaders()
    else:
        headers = response.headers.items()
    for key, value in headers:
        key = str(key.lower())
        if key.startswith('x-image-meta-property-'):
            field_name = key[len('x-image-meta-property-'):].replace('-', '_')
            properties[field_name] = value or None
        elif key.startswith('x-image-meta-'):
            field_name = key[len('x-image-meta-'):].replace('-', '_')
            if 'x-image-meta-' + field_name not in IMAGE_META_HEADERS:
                msg = _('Bad header: %(header_name)s') % {'header_name': key}
                raise exc.HTTPBadRequest(msg, content_type='text/plain')
            result[field_name] = value or None
    result['properties'] = properties
    for key, nullable in [('size', False), ('min_disk', False), ('min_ram', False), ('virtual_size', True)]:
        if key in result:
            try:
                result[key] = int(result[key])
            except ValueError:
                if nullable and result[key] == str(None):
                    result[key] = None
                else:
                    extra = _("Cannot convert image %(key)s '%(value)s' to an integer.") % {'key': key, 'value': result[key]}
                    raise exception.InvalidParameterValue(value=result[key], param=key, extra_msg=extra)
            if result[key] is not None and result[key] < 0:
                extra = _('Cannot be a negative value.')
                raise exception.InvalidParameterValue(value=result[key], param=key, extra_msg=extra)
    for key in ('is_public', 'deleted', 'protected'):
        if key in result:
            result[key] = strutils.bool_from_string(result[key])
    return result