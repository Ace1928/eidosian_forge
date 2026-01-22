from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def byrevrank(self, key, rank, *ranks):
    """
        Retrieve an estimation of the value with the given reverse rank.

        For more information see `TDIGEST.BY_REVRANK <https://redis.io/commands/tdigest.by_revrank>`_.
        """
    return self.execute_command(TDIGEST_BYREVRANK, key, rank, *ranks)