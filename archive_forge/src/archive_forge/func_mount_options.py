from pprint import pformat
from six import iteritems
import re
@mount_options.setter
def mount_options(self, mount_options):
    """
        Sets the mount_options of this V1PersistentVolumeSpec.
        A list of mount options, e.g. ["ro", "soft"]. Not validated - mount
        will simply fail if one is invalid. More info:
        https://kubernetes.io/docs/concepts/storage/persistent-volumes/#mount-options

        :param mount_options: The mount_options of this V1PersistentVolumeSpec.
        :type: list[str]
        """
    self._mount_options = mount_options