from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.compute.drivers.gce import GCEConnection, GCENodeDriver
def ex_balancer_list_healthchecks(self, balancer):
    """
        Return list of healthchecks attached to balancer

        :param  balancer: LoadBalancer which should be used
        :type   balancer: :class:`LoadBalancer`

        :rtype: ``list`` of :class:`HealthChecks`
        """
    return balancer.extra['healthchecks']