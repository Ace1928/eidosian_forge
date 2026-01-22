import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_destroy_node_and_drives(self, node):
    """
        Destroy a node and all the drives associated with it.

        :param      node: Node which should be used
        :type       node: :class:`libcloud.compute.base.Node`

        :rtype: ``bool``
        """
    node = self._get_node_info(node)
    drive_uuids = []
    for key, value in node.items():
        if (key.startswith('ide:') or key.startswith('scsi') or key.startswith('block')) and (not (key.endswith(':bytes') or key.endswith(':requests') or key.endswith('media'))):
            drive_uuids.append(value)
    node_destroyed = self.destroy_node(self._to_node(node))
    if not node_destroyed:
        return False
    for drive_uuid in drive_uuids:
        self.ex_drive_destroy(drive_uuid)
    return True