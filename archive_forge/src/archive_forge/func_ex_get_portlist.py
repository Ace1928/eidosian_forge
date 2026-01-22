import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_portlist(self, ex_portlist_id):
    """
        Get Port List

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # Get specific portlist by ID
        >>> portlist_id = '27dd8c66-80ff-496b-9f54-2a3da2fe679e'
        >>> portlist = driver.ex_get_portlist(portlist_id)
        >>> pprint(portlist)

        :param  ex_portlist_id: The ex_port_list or ex_port_list ID
        :type   ex_portlist_id: :class:`NttCisNetworkDomain` or 'str'

        :return:  NttCisPortList object
        :rtype:  :class:`NttCisPort`
        """
    url_path = 'network/portList/%s' % ex_portlist_id
    response = self.connection.request_with_orgId_api_2(url_path).object
    return self._to_port_list(response)