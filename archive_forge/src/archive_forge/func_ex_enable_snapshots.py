import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_enable_snapshots(self, node, window, plan='ADVANCED', initiate='true'):
    """
        Enable snapshot service on a server

        :param      node: Node ID of the node on which to enable snapshots.
        :type       node: ``str``

        :param      window: The window id of the window in which the
                            snapshot is enabled.
        :type       name: ``str``

        :param      plan: Pland type 'ESSENTIALS' or 'ADVANCED
        :type       plan: ``str``

        :param      initiate: Run a snapshot upon configuration of the
                              snapshot.
        :type       ``str``

        :rtype: ``bool``
        """
    update_node = ET.Element('enableSnapshotService', {'xmlns': TYPES_URN})
    window_id = window
    plan = plan
    update_node.set('serverId', node)
    ET.SubElement(update_node, 'servicePlan').text = plan
    ET.SubElement(update_node, 'windowId').text = window_id
    ET.SubElement(update_node, 'initiateManualSnapshot').text = initiate
    result = self.connection.request_with_orgId_api_2('snapshot/enableSnapshotService', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']