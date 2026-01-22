import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.minimum_version('1.24')
def get_unlock_key(self):
    """
            Get the unlock key for this Swarm manager.

            Returns:
                A ``dict`` containing an ``UnlockKey`` member
        """
    return self._result(self._get(self._url('/swarm/unlockkey')), True)