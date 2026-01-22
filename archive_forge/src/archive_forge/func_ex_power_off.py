import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_power_off(self, node):
    """
        This function will abruptly power-off a server.  Unlike
        ex_shutdown_graceful, success ensures the node will stop but some OS
        and application configurations may be adversely affected by the
        equivalent of pulling the power plug out of the machine.

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
    request_elm = ET.Element('powerOffServer', {'xmlns': TYPES_URN, 'id': node.id})
    try:
        body = self.connection.request_with_orgId_api_2('server/powerOffServer', method='POST', data=ET.tostring(request_elm)).object
        response_code = findtext(body, 'responseCode', TYPES_URN)
    except (NttCisAPIException, NameError, BaseHTTPError):
        r = self.ex_get_node_by_id(node.id)
        response_code = r.state.upper()
    return response_code in ['IN_PROGRESS', 'OK', 'STOPPED', 'STOPPING']