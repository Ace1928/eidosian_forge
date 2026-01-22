from pprint import pformat
from six import iteritems
import re
@fs_type.setter
def fs_type(self, fs_type):
    """
        Sets the fs_type of this V1PortworxVolumeSource.
        FSType represents the filesystem type to mount Must be a filesystem type
        supported by the host operating system. Ex. "ext4", "xfs".
        Implicitly inferred to be "ext4" if unspecified.

        :param fs_type: The fs_type of this V1PortworxVolumeSource.
        :type: str
        """
    self._fs_type = fs_type