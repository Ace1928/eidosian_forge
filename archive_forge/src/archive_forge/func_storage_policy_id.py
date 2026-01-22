from pprint import pformat
from six import iteritems
import re
@storage_policy_id.setter
def storage_policy_id(self, storage_policy_id):
    """
        Sets the storage_policy_id of this V1VsphereVirtualDiskVolumeSource.
        Storage Policy Based Management (SPBM) profile ID associated with the
        StoragePolicyName.

        :param storage_policy_id: The storage_policy_id of this
        V1VsphereVirtualDiskVolumeSource.
        :type: str
        """
    self._storage_policy_id = storage_policy_id