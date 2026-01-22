from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_create_ssl_offload_profile(self, netowrk_domain_id, name, ssl_domain_cert_id, description=None, ciphers=None, ssl_cert_chain_id=None):
    """
        Creates an SSL Offload profile

        :param network_domain_id: The network domain's Id
        :type netowrk_domain_id: ``str``
        :param name: Offload profile's name
        :type name: ``str``
        :param ssl_domain_cert_id: Certificate's Id
        :type ssl_domain_cert_id: ``str``
        :param description: (Optional) Profile's description
        :type description: ``str``
        :param ciphers: (Optional) The default cipher string is:
        "MEDIUM:HIGH:!EXPORT:!ADH:!MD5:!RC4:!SSLv2:!SSLv3:
        !ECDHE+AES-GCM:!ECDHE+AES:!ECDHE+3DES:!ECDHE_ECDSA:
        !ECDH_RSA:!ECDH_ECDSA:@SPEED" It is possible to choose just a subset
        of this string
        :type ciphers: ``str``
        :param ssl_cert_chain_id: (Optional) Bind the certificate
        chain to the profile.
        :type ssl_cert_chain_id: `str``
        :returns: ``bool``
        """
    ssl_offload_elem = ET.Element('createSslOffloadProfile', {'xmlns': TYPES_URN})
    ET.SubElement(ssl_offload_elem, 'networkDomainId').text = netowrk_domain_id
    ET.SubElement(ssl_offload_elem, 'name').text = name
    if description is not None:
        ET.SubElement(ssl_offload_elem, 'description').text = description
    if ciphers is not None:
        ET.SubElement(ssl_offload_elem, 'ciphers').text = ciphers
    ET.SubElement(ssl_offload_elem, 'sslDomainCertificateId').text = ssl_domain_cert_id
    if ssl_cert_chain_id is not None:
        ET.SubElement(ssl_offload_elem, 'sslCertificateChainId').text = ssl_cert_chain_id
    result = self.connection.request_with_orgId_api_2('networkDomainVip/createSslOffloadProfile', method='POST', data=ET.tostring(ssl_offload_elem)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']