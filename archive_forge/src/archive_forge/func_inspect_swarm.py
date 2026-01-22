import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.minimum_version('1.24')
def inspect_swarm(self):
    """
        Retrieve low-level information about the current swarm.

        Returns:
            A dictionary containing data about the swarm.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/swarm')
    return self._result(self._get(url), True)