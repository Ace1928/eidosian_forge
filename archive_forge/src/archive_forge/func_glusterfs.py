from pprint import pformat
from six import iteritems
import re
@glusterfs.setter
def glusterfs(self, glusterfs):
    """
        Sets the glusterfs of this V1PersistentVolumeSpec.
        Glusterfs represents a Glusterfs volume that is attached to a host and
        exposed to the pod. Provisioned by an admin. More info:
        https://releases.k8s.io/HEAD/examples/volumes/glusterfs/README.md

        :param glusterfs: The glusterfs of this V1PersistentVolumeSpec.
        :type: V1GlusterfsPersistentVolumeSource
        """
    self._glusterfs = glusterfs