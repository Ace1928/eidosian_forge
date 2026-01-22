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
def update_volume_connector(self, volume_connector, **attrs):
    """Update a volume_connector.

        :param volume_connector: Either the ID of a volume_connector
            or an instance of
            :class:`~openstack.baremetal.v1.volume_connector.VolumeConnector`.
        :param dict attrs: The attributes to update on the
            volume_connector represented by the ``volume_connector``
            parameter.

        :returns: The updated volume_connector.
        :rtype:
            :class:`~openstack.baremetal.v1.volume_connector.VolumeConnector`
        """
    return self._update(_volumeconnector.VolumeConnector, volume_connector, **attrs)