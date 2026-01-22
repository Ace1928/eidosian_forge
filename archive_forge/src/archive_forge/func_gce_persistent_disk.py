from pprint import pformat
from six import iteritems
import re
@gce_persistent_disk.setter
def gce_persistent_disk(self, gce_persistent_disk):
    """
        Sets the gce_persistent_disk of this V1PersistentVolumeSpec.
        GCEPersistentDisk represents a GCE Disk resource that is attached to a
        kubelet's host machine and then exposed to the pod. Provisioned by an
        admin. More info:
        https://kubernetes.io/docs/concepts/storage/volumes#gcepersistentdisk

        :param gce_persistent_disk: The gce_persistent_disk of this
        V1PersistentVolumeSpec.
        :type: V1GCEPersistentDiskVolumeSource
        """
    self._gce_persistent_disk = gce_persistent_disk