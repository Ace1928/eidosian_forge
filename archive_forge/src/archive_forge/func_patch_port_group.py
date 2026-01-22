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
def patch_port_group(self, port_group, patch):
    """Apply a JSON patch to the port_group.

        :param port_group: The value can be the ID of a port group or a
            :class:`~openstack.baremetal.v1.port_group.PortGroup` instance.
        :param patch: JSON patch to apply.

        :returns: The updated port group.
        :rtype: :class:`~openstack.baremetal.v1.port_group.PortGroup`
        """
    res = self._get_resource(_portgroup.PortGroup, port_group)
    return res.patch(self, patch)