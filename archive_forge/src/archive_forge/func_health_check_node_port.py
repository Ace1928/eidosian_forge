from pprint import pformat
from six import iteritems
import re
@health_check_node_port.setter
def health_check_node_port(self, health_check_node_port):
    """
        Sets the health_check_node_port of this V1ServiceSpec.
        healthCheckNodePort specifies the healthcheck nodePort for the service.
        If not specified, HealthCheckNodePort is created by the service api
        backend with the allocated nodePort. Will use user-specified nodePort
        value if specified by the client. Only effects when Type is set to
        LoadBalancer and ExternalTrafficPolicy is set to Local.

        :param health_check_node_port: The health_check_node_port of this
        V1ServiceSpec.
        :type: int
        """
    self._health_check_node_port = health_check_node_port