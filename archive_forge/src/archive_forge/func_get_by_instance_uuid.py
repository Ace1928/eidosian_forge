import logging
import os
from oslo_utils import strutils
from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def get_by_instance_uuid(self, instance_uuid, fields=None, os_ironic_api_version=None, global_request_id=None):
    path = '?instance_uuid=%s' % instance_uuid
    if fields is not None:
        path += '&fields=' + ','.join(fields)
    else:
        path = 'detail' + path
    nodes = self._list(self._path(path), 'nodes', os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)
    if len(nodes) == 1:
        return nodes[0]
    else:
        raise exc.NotFound()