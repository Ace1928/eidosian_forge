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
def remove_node_trait(self, node, trait, ignore_missing=True):
    """Remove a trait from a node.

        :param node: The value can be the name or ID of a node or a
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :param trait: trait to remove from the node.
        :param bool ignore_missing: When set to ``False``, an exception
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the trait could not be found. When set to ``True``, no
            exception will be raised when attempting to delete a non-existent
            trait.
        :returns: The updated :class:`~openstack.baremetal.v1.node.Node`
        """
    res = self._get_resource(_node.Node, node)
    return res.remove_trait(self, trait, ignore_missing=ignore_missing)