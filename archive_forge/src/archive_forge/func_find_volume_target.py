from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def find_volume_target(self, vt_id, ignore_missing=True):
    """Find a single volume target.

        :param str vt_id: The ID of a volume target.

        :param bool ignore_missing: When set to ``False``, an exception of
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the volume connector does not exist.  When set to `True``,
            None will be returned when attempting to find a nonexistent
            volume target.
        :returns: One
            :class:`~openstack.baremetal.v1.volumetarget.VolumeTarget`
            object or None.
        """
    return self._find(_volumetarget.VolumeTarget, vt_id, ignore_missing=ignore_missing)