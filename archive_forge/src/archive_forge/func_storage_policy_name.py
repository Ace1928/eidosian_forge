from pprint import pformat
from six import iteritems
import re
@storage_policy_name.setter
def storage_policy_name(self, storage_policy_name):
    """
        Sets the storage_policy_name of this V1VsphereVirtualDiskVolumeSource.
        Storage Policy Based Management (SPBM) profile name.

        :param storage_policy_name: The storage_policy_name of this
        V1VsphereVirtualDiskVolumeSource.
        :type: str
        """
    self._storage_policy_name = storage_policy_name