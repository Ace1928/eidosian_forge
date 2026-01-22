from pprint import pformat
from six import iteritems
import re
@dns_config.setter
def dns_config(self, dns_config):
    """
        Sets the dns_config of this V1PodSpec.
        Specifies the DNS parameters of a pod. Parameters specified here will be
        merged to the generated DNS configuration based on DNSPolicy.

        :param dns_config: The dns_config of this V1PodSpec.
        :type: V1PodDNSConfig
        """
    self._dns_config = dns_config