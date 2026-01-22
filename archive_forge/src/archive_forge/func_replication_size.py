import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def replication_size(options, args):
    """%(prog)s size <server:port>

    Determine the size of a glance instance if dumped to disk.

    server:port: the location of the glance instance.
    """
    if args is None or len(args) < 1:
        raise TypeError(_('Too few arguments.'))
    server, port = utils.parse_valid_host_port(args.pop())
    total_size = 0
    count = 0
    imageservice = get_image_service()
    client = imageservice(http.HTTPConnection(server, port), options.targettoken)
    for image in client.get_images():
        LOG.debug('Considering image: %(image)s', {'image': image})
        if image['status'] == 'active':
            total_size += int(image['size'])
            count += 1
    print(_('Total size is %(size)d bytes (%(human_size)s) across %(img_count)d images') % {'size': total_size, 'human_size': _human_readable_size(total_size), 'img_count': count})