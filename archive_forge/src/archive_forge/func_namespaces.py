from pprint import pformat
from six import iteritems
import re
@namespaces.setter
def namespaces(self, namespaces):
    """
        Sets the namespaces of this V1PodAffinityTerm.
        namespaces specifies which namespaces the labelSelector applies to
        (matches against); null or empty list means "this pod's namespace"

        :param namespaces: The namespaces of this V1PodAffinityTerm.
        :type: list[str]
        """
    self._namespaces = namespaces