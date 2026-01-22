from pprint import pformat
from six import iteritems
import re
@cephfs.setter
def cephfs(self, cephfs):
    """
        Sets the cephfs of this V1PersistentVolumeSpec.
        CephFS represents a Ceph FS mount on the host that shares a pod's
        lifetime

        :param cephfs: The cephfs of this V1PersistentVolumeSpec.
        :type: V1CephFSPersistentVolumeSource
        """
    self._cephfs = cephfs