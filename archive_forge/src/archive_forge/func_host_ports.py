from pprint import pformat
from six import iteritems
import re
@host_ports.setter
def host_ports(self, host_ports):
    """
        Sets the host_ports of this PolicyV1beta1PodSecurityPolicySpec.
        hostPorts determines which host port ranges are allowed to be exposed.

        :param host_ports: The host_ports of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[PolicyV1beta1HostPortRange]
        """
    self._host_ports = host_ports