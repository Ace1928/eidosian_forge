from pprint import pformat
from six import iteritems
import re
@load_balancer.setter
def load_balancer(self, load_balancer):
    """
        Sets the load_balancer of this ExtensionsV1beta1IngressStatus.
        LoadBalancer contains the current status of the load-balancer.

        :param load_balancer: The load_balancer of this
        ExtensionsV1beta1IngressStatus.
        :type: V1LoadBalancerStatus
        """
    self._load_balancer = load_balancer