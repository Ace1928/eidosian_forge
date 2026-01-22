import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_change_storage_size(self, disk_id, size):
    """
        Change the size of a disk

        :param  node: The server to change the disk of
        :type   node: :class:`Node`

        :param  disk_id: The ID of the disk to resize
        :type   disk_id: ``str``

        :param  size: The disk size in GB
        :type   size: ``int``

        :rtype: ``bool``
        """
    create_node = ET.Element('expandDisk', {'xmlns': TYPES_URN, 'id': disk_id})
    ET.SubElement(create_node, 'newSizeGb').text = str(size)
    "\n        This code is for version 1 of MCP, need version 2\n        result = self.connection.request_with_orgId_api_1(\n            'server/%s/disk/%s/changeSize' %\n            (node.id, disk_id),\n            method='POST',\n            data=ET.tostring(create_node)).object\n        "
    result = self.connection.request_with_orgId_api_2('server/expandDisk', method='POST', data=ET.tostring(create_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']