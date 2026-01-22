from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def destroy_balancer(self, balancer):
    self._async_request(command='deleteLoadBalancerRule', params={'id': balancer.id}, method='GET')
    self._async_request(command='disassociateIpAddress', params={'id': balancer.ex_public_ip_id}, method='GET')