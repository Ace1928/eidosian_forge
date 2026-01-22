from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import core
from zaqarclient.queues.v1 import flavor
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import pool
from zaqarclient.queues.v1 import queues
from zaqarclient import transport
from zaqarclient.transport import errors
from zaqarclient.transport import request
@decorators.version(min_version=1.1)
def flavors(self, **params):
    """Gets a list of flavors from the server

        :param params: Filters to use for getting flavors
        :type params: dict.

        :returns: A list of flavors
        :rtype: `list`
        """
    req, trans = self._request_and_transport()
    flavor_list = core.flavor_list(trans, req, **params)
    return iterator._Iterator(self, flavor_list, 'flavors', flavor.create_object(self))