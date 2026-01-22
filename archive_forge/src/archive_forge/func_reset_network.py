import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def reset_network(self, server):
    """
        Reset network of an instance.

        :param server: The :class:`Server` for network is to be reset
        :returns: An instance of novaclient.base.TupleWithMeta
        """
    return self._action('resetNetwork', server)