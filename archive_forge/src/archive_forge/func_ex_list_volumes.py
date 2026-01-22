import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
def ex_list_volumes(self, node, disk_id=None):
    """
        List existing disk volumes for for given Linode.

        :keyword    node: Node to list disk volumes for. (required)
        :type       node: :class:`Node`

        :keyword    disk_id: Id for specific disk volume. (optional)
        :type       disk_id: ``int``

        :rtype: ``list`` of :class:`StorageVolume`
        """
    if not isinstance(node, Node):
        raise LinodeException(253, 'Invalid node instance')
    params = {'api_action': 'linode.disk.list', 'LinodeID': node.id}
    if disk_id is not None:
        params['DiskID'] = disk_id
    data = self.connection.request(API_ROOT, params=params).objects[0]
    return self._to_volumes(data)