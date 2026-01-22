from pprint import pformat
from six import iteritems
import re
@vsphere_volume.setter
def vsphere_volume(self, vsphere_volume):
    """
        Sets the vsphere_volume of this V1PersistentVolumeSpec.
        VsphereVolume represents a vSphere volume attached and mounted on
        kubelets host machine

        :param vsphere_volume: The vsphere_volume of this
        V1PersistentVolumeSpec.
        :type: V1VsphereVirtualDiskVolumeSource
        """
    self._vsphere_volume = vsphere_volume