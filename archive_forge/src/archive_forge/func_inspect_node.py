import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.check_resource('node_id')
@utils.minimum_version('1.24')
def inspect_node(self, node_id):
    """
        Retrieve low-level information about a swarm node

        Args:
            node_id (string): ID of the node to be inspected.

        Returns:
            A dictionary containing data about this node.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/nodes/{0}', node_id)
    return self._result(self._get(url), True)