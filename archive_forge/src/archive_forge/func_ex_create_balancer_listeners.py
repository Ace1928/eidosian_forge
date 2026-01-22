from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_create_balancer_listeners(self, name, listeners=None):
    """
        Creates one or more listeners on a load balancer for the specified port

        :param name: The mnemonic name associated with the load balancer
        :type  name: ``str``

        :param listeners: Each tuple contain values, (LoadBalancerPortNumber,
                          InstancePortNumber, Protocol,[SSLCertificateId])
        :type  listeners: ``list of tuple`
        """
    params = {'Action': 'CreateLoadBalancerListeners', 'LoadBalancerName': name}
    for index, listener in enumerate(listeners):
        i = index + 1
        protocol = listener[2].upper()
        params['Listeners.member.%d.LoadBalancerPort' % i] = listener[0]
        params['Listeners.member.%d.InstancePort' % i] = listener[1]
        params['Listeners.member.%d.Protocol' % i] = listener[2]
        if protocol == 'HTTPS' or protocol == 'SSL':
            params['Listeners.member.%d.                           SSLCertificateId' % i] = listener[3]
    else:
        return False
    response = self.connection.request(ROOT, params=params)
    return response.status == httplib.OK