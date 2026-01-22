from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_stop_listener(self, balancer, port):
    """
        Stop balancer's listener listening the given port.

        :param balancer: a load balancer
        :type balancer: ``LoadBalancer``

        :param port: listening port
        :type port: ``int``

        :return: whether operation is success
        :rtype: ``bool``
        """
    params = {'Action': 'StopLoadBalancerListener', 'LoadBalancerId': balancer.id, 'ListenerPort': port}
    resp = self.connection.request(self.path, params)
    return resp.success()