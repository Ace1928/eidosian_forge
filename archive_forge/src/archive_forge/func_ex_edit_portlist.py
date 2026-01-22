import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_edit_portlist(self, ex_portlist, description=None, port_collection=None, child_portlist_list=None):
    """
        Edit Port List.

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> from libcloud.common.NTTCIS import DimensionDataPort
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # Port Collection
        >>> port_1 = DimensionDataPort(begin='4200')
        >>> port_2 = DimensionDataPort(begin='4201', end='4210')
        >>> port_collection = [port_1, port_2]
        >>>
        >>> # Edit Port List
        >>> editPortlist = driver.ex_get_portlist(
            '27dd8c66-80ff-496b-9f54-2a3da2fe679e')
        >>>
        >>> result = driver.ex_edit_portlist(
        >>>     ex_portlist=editPortlist.id,
        >>>     description="Make Changes in portlist",
        >>>     port_collection=port_collection,
        >>>     child_portlist_list={'a9cd4984-6ff5-4f93-89ff-8618ab642bb9'}
        >>> )
        >>> pprint(result)

        :param    ex_portlist:  Port List to be edited
                                        (required)
        :type      ex_portlist: :``str`` or :class:'DNttCisPortList'

        :param    description:  Port List Description
        :type      description: :``str``

        :param    port_collection:  List of Ports
        :type      port_collection: :``str``

        :param    child_portlist_list:  Child PortList to be included in
                                          this IP Address List
        :type      child_portlist_list: :``list`` of
                                          :class'NttCisChildPortList'
                                          or ''str''

        :return: a list of NttCisPortList objects
        :rtype: ``list`` of :class:`NttCisPortList`
        """
    existing_port_address_list = ET.Element('editPortList', {'id': self._port_list_to_port_list_id(ex_portlist), 'xmlns': TYPES_URN, 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
    if description is not None:
        if description != 'nil':
            ET.SubElement(existing_port_address_list, 'description').text = description
        else:
            ET.SubElement(existing_port_address_list, 'description', {'xsi:nil': 'true'})
    if port_collection is not None:
        for port in port_collection:
            p = ET.SubElement(existing_port_address_list, 'port')
            p.set('begin', port.begin)
            if port.end:
                p.set('end', port.end)
    if child_portlist_list is not None:
        for child in child_portlist_list:
            ET.SubElement(existing_port_address_list, 'childPortListId').text = self._child_port_list_to_child_port_list_id(child)
    else:
        ET.SubElement(existing_port_address_list, 'childPortListId', {'xsi:nil': 'true'})
    response = self.connection.request_with_orgId_api_2('network/editPortList', method='POST', data=ET.tostring(existing_port_address_list)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']