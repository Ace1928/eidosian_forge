from pprint import pformat
from six import iteritems
import re
@volume_path.setter
def volume_path(self, volume_path):
    """
        Sets the volume_path of this V1VsphereVirtualDiskVolumeSource.
        Path that identifies vSphere volume vmdk

        :param volume_path: The volume_path of this
        V1VsphereVirtualDiskVolumeSource.
        :type: str
        """
    if volume_path is None:
        raise ValueError('Invalid value for `volume_path`, must not be `None`')
    self._volume_path = volume_path