from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def byrank(self, key, rank, *ranks):
    """
        Retrieve an estimation of the value with the given rank.

        For more information see `TDIGEST.BY_RANK <https://redis.io/commands/tdigest.by_rank>`_.
        """
    return self.execute_command(TDIGEST_BYRANK, key, rank, *ranks)