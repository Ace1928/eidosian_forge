from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
@get_params
def ex_list_ssl_domain_certs(self, params={}):
    """
        Functions takes a named parameter that can be one or none of the
        following

        :param params: A sequence of comma separated keyword arguments
        and a value
            * id=
            * network_domain_id=
            * name=
            * state=
            * create_time=
            * expiry_time=
        :returns: `list` of :class: `NttCisDomaincertificate`
        """
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/sslDomainCertificate', params=params, method='GET').object
    return self._to_certs(result)