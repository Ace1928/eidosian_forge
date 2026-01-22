from pprint import pformat
from six import iteritems
import re
@mount_propagation.setter
def mount_propagation(self, mount_propagation):
    """
        Sets the mount_propagation of this V1VolumeMount.
        mountPropagation determines how mounts are propagated from the host to
        container and the other way around. When not set, MountPropagationNone
        is used. This field is beta in 1.10.

        :param mount_propagation: The mount_propagation of this V1VolumeMount.
        :type: str
        """
    self._mount_propagation = mount_propagation