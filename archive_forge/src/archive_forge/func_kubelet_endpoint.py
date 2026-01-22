from pprint import pformat
from six import iteritems
import re
@kubelet_endpoint.setter
def kubelet_endpoint(self, kubelet_endpoint):
    """
        Sets the kubelet_endpoint of this V1NodeDaemonEndpoints.
        Endpoint on which Kubelet is listening.

        :param kubelet_endpoint: The kubelet_endpoint of this
        V1NodeDaemonEndpoints.
        :type: V1DaemonEndpoint
        """
    self._kubelet_endpoint = kubelet_endpoint