from pprint import pformat
from six import iteritems
import re
@external_i_ps.setter
def external_i_ps(self, external_i_ps):
    """
        Sets the external_i_ps of this V1ServiceSpec.
        externalIPs is a list of IP addresses for which nodes in the cluster
        will also accept traffic for this service.  These IPs are not managed by
        Kubernetes.  The user is responsible for ensuring that traffic arrives
        at a node with this IP.  A common example is external load-balancers
        that are not part of the Kubernetes system.

        :param external_i_ps: The external_i_ps of this V1ServiceSpec.
        :type: list[str]
        """
    self._external_i_ps = external_i_ps