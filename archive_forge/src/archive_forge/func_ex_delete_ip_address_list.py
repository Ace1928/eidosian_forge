import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_ip_address_list(self, ex_ip_address_list):
    """
        Delete IP Address List by ID

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> ip_address_list_id = '5e7c323f-c885-4e4b-9a27-94c44217dbd3'
        >>> result = driver.ex_delete_ip_address_list(ip_address_list_id)
        >>> pprint(result)

        :param    ex_ip_address_list:  IP Address List object or IP Address
                                        List ID (required)
        :type     ex_ip_address_list: :class:'NttCisIpAddressList'
                    or ``str``

        :rtype: ``bool``
        """
    delete_ip_address_list = ET.Element('deleteIpAddressList', {'xmlns': TYPES_URN, 'id': self._ip_address_list_to_ip_address_list_id(ex_ip_address_list)})
    response = self.connection.request_with_orgId_api_2('network/deleteIpAddressList', method='POST', data=ET.tostring(delete_ip_address_list)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']