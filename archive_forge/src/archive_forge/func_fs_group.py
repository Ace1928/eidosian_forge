from pprint import pformat
from six import iteritems
import re
@fs_group.setter
def fs_group(self, fs_group):
    """
        Sets the fs_group of this V1PodSecurityContext.
        A special supplemental group that applies to all containers in a pod.
        Some volume types allow the Kubelet to change the ownership of that
        volume to be owned by the pod:  1. The owning GID will be the FSGroup 2.
        The setgid bit is set (new files created in the volume will be owned by
        FSGroup) 3. The permission bits are OR'd with rw-rw----  If unset, the
        Kubelet will not modify the ownership and permissions of any volume.

        :param fs_group: The fs_group of this V1PodSecurityContext.
        :type: int
        """
    self._fs_group = fs_group