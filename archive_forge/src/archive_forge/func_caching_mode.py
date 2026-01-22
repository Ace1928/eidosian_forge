from pprint import pformat
from six import iteritems
import re
@caching_mode.setter
def caching_mode(self, caching_mode):
    """
        Sets the caching_mode of this V1AzureDiskVolumeSource.
        Host Caching mode: None, Read Only, Read Write.

        :param caching_mode: The caching_mode of this V1AzureDiskVolumeSource.
        :type: str
        """
    self._caching_mode = caching_mode