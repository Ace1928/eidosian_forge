from pprint import pformat
from six import iteritems
import re
@load_balancer_source_ranges.setter
def load_balancer_source_ranges(self, load_balancer_source_ranges):
    """
        Sets the load_balancer_source_ranges of this V1ServiceSpec.
        If specified and supported by the platform, this will restrict traffic
        through the cloud-provider load-balancer will be restricted to the
        specified client IPs. This field will be ignored if the cloud-provider
        does not support the feature." More info:
        https://kubernetes.io/docs/tasks/access-application-cluster/configure-cloud-provider-firewall/

        :param load_balancer_source_ranges: The load_balancer_source_ranges of
        this V1ServiceSpec.
        :type: list[str]
        """
    self._load_balancer_source_ranges = load_balancer_source_ranges