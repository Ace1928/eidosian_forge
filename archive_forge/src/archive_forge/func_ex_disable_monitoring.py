import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_disable_monitoring(self, node):
    """
        Disables cloud monitoring for a node

        :param   node: The node to stop monitoring
        :type    node: :class:`Node`

        :rtype: ``bool``
        """
    update_node = ET.Element('disableServerMonitoring', {'xmlns': TYPES_URN})
    update_node.set('id', node.id)
    result = self.connection.request_with_orgId_api_2('server/disableServerMonitoring', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']