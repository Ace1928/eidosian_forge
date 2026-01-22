import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_attach_node_to_vlan(self, node, vlan=None, private_ipv4=None):
    """
        Attach a node to a VLAN by adding an additional NIC to
        the node on the target VLAN. The IP will be automatically
        assigned based on the VLAN IP network space. Alternatively, provide
        a private IPv4 address instead of VLAN information, and this will
        be assigned to the node on corresponding NIC.

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :param      vlan: VLAN to attach the node to
                          (required unless private_ipv4)
        :type       vlan: :class:`NttCisVlan`

        :keyword    private_ipv4: Private nic IPv4 Address
                                  (required unless vlan)
        :type       private_ipv4: ``str``

        :rtype: ``bool``
        """
    request = ET.Element('addNic', {'xmlns': TYPES_URN})
    ET.SubElement(request, 'serverId').text = node.id
    nic = ET.SubElement(request, 'nic')
    if vlan is not None:
        ET.SubElement(nic, 'vlanId').text = vlan.id
    elif private_ipv4 is not None:
        ET.SubElement(nic, 'privateIpv4').text = private_ipv4
    else:
        raise ValueError('One of vlan or primary_ipv4 must be specified')
    response = self.connection.request_with_orgId_api_2('server/addNic', method='POST', data=ET.tostring(request)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']