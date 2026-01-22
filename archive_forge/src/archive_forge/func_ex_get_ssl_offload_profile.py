from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_ssl_offload_profile(self, profile_id):
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/sslOffloadProfile/%s' % profile_id, method='GET').object
    return self._to_ssl_profile(result)