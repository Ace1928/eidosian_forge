from pprint import pformat
from six import iteritems
import re
@rbd.setter
def rbd(self, rbd):
    """
        Sets the rbd of this V1PersistentVolumeSpec.
        RBD represents a Rados Block Device mount on the host that shares a
        pod's lifetime. More info:
        https://releases.k8s.io/HEAD/examples/volumes/rbd/README.md

        :param rbd: The rbd of this V1PersistentVolumeSpec.
        :type: V1RBDPersistentVolumeSource
        """
    self._rbd = rbd