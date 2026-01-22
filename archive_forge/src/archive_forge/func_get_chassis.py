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
def get_chassis(self, chassis, fields=None):
    """Get a specific chassis.

        :param chassis: The value can be the ID of a chassis or a
            :class:`~openstack.baremetal.v1.chassis.Chassis` instance.
        :param fields: Limit the resource fields to fetch.

        :returns: One :class:`~openstack.baremetal.v1.chassis.Chassis`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            chassis matching the name or ID could be found.
        """
    return self._get_with_fields(_chassis.Chassis, chassis, fields=fields)