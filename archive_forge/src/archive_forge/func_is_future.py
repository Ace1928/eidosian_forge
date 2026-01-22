from collections import UserDict
from dask.distributed import wait
from distributed import Future
from distributed.client import default_client
@classmethod
def is_future(cls, item):
    """
        Check if the item is a Future.

        Parameters
        ----------
        item : distributed.Future or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
    return isinstance(item, Future)