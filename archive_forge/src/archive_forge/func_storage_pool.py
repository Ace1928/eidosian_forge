from pprint import pformat
from six import iteritems
import re
@storage_pool.setter
def storage_pool(self, storage_pool):
    """
        Sets the storage_pool of this V1ScaleIOVolumeSource.
        The ScaleIO Storage Pool associated with the protection domain.

        :param storage_pool: The storage_pool of this V1ScaleIOVolumeSource.
        :type: str
        """
    self._storage_pool = storage_pool