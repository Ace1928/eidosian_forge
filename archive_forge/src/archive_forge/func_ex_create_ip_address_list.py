import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_ip_address_list(self, ex_network_domain, name, description, ip_version, ip_address_collection, child_ip_address_list=None):
    """
        Create IP Address List. IP Address list.

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> from libcloud.common.nttcis import NttCisIpAddress
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # Get location
        >>> location = driver.ex_get_location_by_id(id='AU9')
        >>>
        >>> # Get network domain by location
        >>> networkDomainName = "Baas QA"
        >>> network_domains = driver.ex_list_network_domains(location=location)
        >>> my_network_domain = [d for d in network_domains if d.name ==
                              networkDomainName][0]
        >>>
        >>> # IP Address collection
        >>> ipAddress_1 = NttCisIpAddress(begin='190.2.2.100')
        >>> ipAddress_2 = NttCisIpAddress(begin='190.2.2.106',
                                                 end='190.2.2.108')
        >>> ipAddress_3 = NttCisIpAddress(begin='190.2.2.0',
                                                 prefix_size='24')
        >>> ip_address_collection = [ipAddress_1, ipAddress_2, ipAddress_3]
        >>>
        >>> # Create IPAddressList
        >>> result = driver.ex_create_ip_address_list(
        >>>     ex_network_domain=my_network_domain,
        >>>     name='My_IP_AddressList_2',
        >>>     ip_version='IPV4',
        >>>     description='Test only',
        >>>     ip_address_collection=ip_address_collection,
        >>>     child_ip_address_list='08468e26-eeb3-4c3d-8ff2-5351fa6d8a04'
        >>> )
        >>>
        >>> pprint(result)


        :param  ex_network_domain: The network domain or network domain ID
        :type   ex_network_domain: :class:`NttCisNetworkDomain` or 'str'

        :param    name:  IP Address List Name (required)
        :type      name: :``str``

        :param    description:  IP Address List Description (optional)
        :type      description: :``str``

        :param    ip_version:  IP Version of ip address (required)
        :type      ip_version: :``str``

        :param    ip_address_collection:  List of IP Address. At least one
                                          ipAddress element or one
                                          childIpAddressListId element must
                                          be provided.
        :type      ip_address_collection: :``str``

        :param    child_ip_address_list:  Child IP Address List or id to be
                                          included in this IP Address List.
                                          At least one ipAddress or
                                          one childIpAddressListId
                                          must be provided.
        :type     child_ip_address_list:
                        :class:'NttCisChildIpAddressList` or `str``

        :return: a list of NttCisIpAddressList objects
        :rtype: ``list`` of :class:`NttCisIpAddressList`
        """
    if ip_address_collection is None and child_ip_address_list is None:
        raise ValueError('At least one ipAddress element or one childIpAddressListId element must be provided.')
    create_ip_address_list = ET.Element('createIpAddressList', {'xmlns': TYPES_URN})
    ET.SubElement(create_ip_address_list, 'networkDomainId').text = self._network_domain_to_network_domain_id(ex_network_domain)
    ET.SubElement(create_ip_address_list, 'name').text = name
    ET.SubElement(create_ip_address_list, 'description').text = description
    ET.SubElement(create_ip_address_list, 'ipVersion').text = ip_version
    for ip in ip_address_collection:
        ip_address = ET.SubElement(create_ip_address_list, 'ipAddress')
        ip_address.set('begin', ip.begin)
        if ip.end:
            ip_address.set('end', ip.end)
        if ip.prefix_size:
            ip_address.set('prefixSize', ip.prefix_size)
    if child_ip_address_list is not None:
        ET.SubElement(create_ip_address_list, 'childIpAddressListId').text = self._child_ip_address_list_to_child_ip_address_list_id(child_ip_address_list)
    response = self.connection.request_with_orgId_api_2('network/createIpAddressList', method='POST', data=ET.tostring(create_ip_address_list)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']