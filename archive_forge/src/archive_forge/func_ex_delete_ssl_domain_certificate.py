from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_delete_ssl_domain_certificate(self, dom_cert_id):
    """
        Deletes an SSL domain certificate

        :param dom_cert_id: Id of certificate to delete
        :type dom_cert_id: ``str``
        :return: ``bool``
        """
    del_dom_cert_elem = ET.Element('deleteSslDomainCertificate', {'id': dom_cert_id, 'xmlns': TYPES_URN})
    result = self.connection.request_with_orgId_api_2('networkDomainVip/deleteSslDomainCertificate', method='POST', data=ET.tostring(del_dom_cert_elem)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']