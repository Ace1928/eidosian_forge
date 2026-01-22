from pprint import pformat
from six import iteritems
import re
@photon_persistent_disk.setter
def photon_persistent_disk(self, photon_persistent_disk):
    """
        Sets the photon_persistent_disk of this V1PersistentVolumeSpec.
        PhotonPersistentDisk represents a PhotonController persistent disk
        attached and mounted on kubelets host machine

        :param photon_persistent_disk: The photon_persistent_disk of this
        V1PersistentVolumeSpec.
        :type: V1PhotonPersistentDiskVolumeSource
        """
    self._photon_persistent_disk = photon_persistent_disk