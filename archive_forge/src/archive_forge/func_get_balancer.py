from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def get_balancer(self, balancer_id):
    balancer = self._sync_request(command='listLoadBalancerRules', params={'id': balancer_id}, method='GET')
    balancer = balancer.get('loadbalancerrule', [])
    if not balancer:
        raise Exception('no such load balancer: ' + str(balancer_id))
    return self._to_balancer(balancer[0])