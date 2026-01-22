from pprint import pformat
from six import iteritems
import re
@host_ip.setter
def host_ip(self, host_ip):
    """
        Sets the host_ip of this V1ContainerPort.
        What host IP to bind the external port to.

        :param host_ip: The host_ip of this V1ContainerPort.
        :type: str
        """
    self._host_ip = host_ip