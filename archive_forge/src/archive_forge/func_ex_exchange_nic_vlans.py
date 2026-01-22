import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_exchange_nic_vlans(self, nic_id_1, nic_id_2):
    """
        Exchange NIC Vlans

        :param    nic_id_1:  Nic ID 1
        :type     nic_id_1: :``str``

        :param    nic_id_2:  Nic ID 2
        :type     nic_id_2: :``str``

        :rtype: ``bool``
        """
    exchange_elem = ET.Element('urn:exchangeNicVlans', {'xmlns:urn': TYPES_URN})
    ET.SubElement(exchange_elem, 'urn:nicId1').text = nic_id_1
    ET.SubElement(exchange_elem, 'urn:nicId2').text = nic_id_2
    response = self.connection.request_with_orgId_api_2('server/exchangeNicVlans', method='POST', data=ET.tostring(exchange_elem)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']