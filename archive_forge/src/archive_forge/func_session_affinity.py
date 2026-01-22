from pprint import pformat
from six import iteritems
import re
@session_affinity.setter
def session_affinity(self, session_affinity):
    """
        Sets the session_affinity of this V1ServiceSpec.
        Supports "ClientIP" and "None". Used to maintain session affinity.
        Enable client IP based session affinity. Must be ClientIP or None.
        Defaults to None. More info:
        https://kubernetes.io/docs/concepts/services-networking/service/#virtual-ips-and-service-proxies

        :param session_affinity: The session_affinity of this V1ServiceSpec.
        :type: str
        """
    self._session_affinity = session_affinity