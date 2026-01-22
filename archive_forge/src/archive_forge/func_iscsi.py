from pprint import pformat
from six import iteritems
import re
@iscsi.setter
def iscsi(self, iscsi):
    """
        Sets the iscsi of this V1PersistentVolumeSpec.
        ISCSI represents an ISCSI Disk resource that is attached to a kubelet's
        host machine and then exposed to the pod. Provisioned by an admin.

        :param iscsi: The iscsi of this V1PersistentVolumeSpec.
        :type: V1ISCSIPersistentVolumeSource
        """
    self._iscsi = iscsi