import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_update_vm_tools(self, node):
    """
        This function triggers an update of the VMware Tools
        software running on the guest OS of a Server.

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
    request_elm = ET.Element('updateVmwareTools', {'xmlns': TYPES_URN, 'id': node.id})
    body = self.connection.request_with_orgId_api_2('server/updateVmwareTools', method='POST', data=ET.tostring(request_elm)).object
    response_code = findtext(body, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']