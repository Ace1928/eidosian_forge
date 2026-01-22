from pprint import pformat
from six import iteritems
import re
@searches.setter
def searches(self, searches):
    """
        Sets the searches of this V1PodDNSConfig.
        A list of DNS search domains for host-name lookup. This will be appended
        to the base search paths generated from DNSPolicy. Duplicated search
        paths will be removed.

        :param searches: The searches of this V1PodDNSConfig.
        :type: list[str]
        """
    self._searches = searches