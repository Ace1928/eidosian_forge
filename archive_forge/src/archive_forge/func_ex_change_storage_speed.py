import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_change_storage_speed(self, disk_id, speed, iops=None):
    """
        Change the speed (disk tier) of a disk

        :param  node: The server to change the disk speed of
        :type   node: :class:`Node`

        :param  disk_id: The ID of the disk to change
        :type   disk_id: ``str``

        :param  speed: The disk speed type e.g. STANDARD
        :type   speed: ``str``

        :rtype: ``bool``
        """
    create_node = ET.Element('changeDiskSpeed', {'xmlns': TYPES_URN})
    create_node.set('id', disk_id)
    ET.SubElement(create_node, 'speed').text = speed
    if iops is not None:
        ET.SubElement(create_node, 'iops').text = str(iops)
    result = self.connection.request_with_orgId_api_2('server/changeDiskSpeed', method='POST', data=ET.tostring(create_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'SUCCESS']