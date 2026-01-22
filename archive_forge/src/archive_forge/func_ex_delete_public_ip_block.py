import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_public_ip_block(self, block):
    delete_node = ET.Element('removePublicIpBlock', {'xmlns': TYPES_URN})
    delete_node.set('id', block.id)
    result = self.connection.request_with_orgId_api_2('network/removePublicIpBlock', method='POST', data=ET.tostring(delete_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']