from pprint import pformat
from six import iteritems
import re
@volume_handle.setter
def volume_handle(self, volume_handle):
    """
        Sets the volume_handle of this V1CSIPersistentVolumeSource.
        VolumeHandle is the unique volume name returned by the CSI volume
        plugin’s CreateVolume to refer to the volume on all subsequent calls.
        Required.

        :param volume_handle: The volume_handle of this
        V1CSIPersistentVolumeSource.
        :type: str
        """
    if volume_handle is None:
        raise ValueError('Invalid value for `volume_handle`, must not be `None`')
    self._volume_handle = volume_handle