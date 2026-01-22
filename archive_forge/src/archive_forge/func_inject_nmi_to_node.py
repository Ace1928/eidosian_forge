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
def inject_nmi_to_node(self, node):
    """Inject NMI to node.

        Injects a non-maskable interrupt (NMI) message to the node. This is
        used when response time is critical, such as during non-recoverable
        hardware errors. In addition, virsh inject-nmi is useful for triggering
        a crashdump in Windows guests.

        :param node: The value can be the name or ID of a node or a
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :return: None
        """
    res = self._get_resource(_node.Node, node)
    res.inject_nmi(self)