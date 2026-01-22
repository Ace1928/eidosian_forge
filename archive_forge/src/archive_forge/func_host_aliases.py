from pprint import pformat
from six import iteritems
import re
@host_aliases.setter
def host_aliases(self, host_aliases):
    """
        Sets the host_aliases of this V1PodSpec.
        HostAliases is an optional list of hosts and IPs that will be injected
        into the pod's hosts file if specified. This is only valid for
        non-hostNetwork pods.

        :param host_aliases: The host_aliases of this V1PodSpec.
        :type: list[V1HostAlias]
        """
    self._host_aliases = host_aliases