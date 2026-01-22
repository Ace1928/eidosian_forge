from pprint import pformat
from six import iteritems
import re
@dns_policy.setter
def dns_policy(self, dns_policy):
    """
        Sets the dns_policy of this V1PodSpec.
        Set DNS policy for the pod. Defaults to "ClusterFirst". Valid values
        are 'ClusterFirstWithHostNet', 'ClusterFirst', 'Default' or 'None'. DNS
        parameters given in DNSConfig will be merged with the policy selected
        with DNSPolicy. To have DNS options set along with hostNetwork, you have
        to specify DNS policy explicitly to 'ClusterFirstWithHostNet'.

        :param dns_policy: The dns_policy of this V1PodSpec.
        :type: str
        """
    self._dns_policy = dns_policy