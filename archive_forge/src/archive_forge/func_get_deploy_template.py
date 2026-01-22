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
def get_deploy_template(self, deploy_template, fields=None):
    """Get a specific deployment template.

        :param deploy_template: The value can be the name or ID
            of a deployment template
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`
            instance.

        :param fields: Limit the resource fields to fetch.

        :returns: One
            :class:`~openstack.baremetal.v1.deploy_templates.DeployTemplate`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no deployment template matching the name or
            ID could be found.
        """
    return self._get_with_fields(_deploytemplates.DeployTemplate, deploy_template, fields=fields)