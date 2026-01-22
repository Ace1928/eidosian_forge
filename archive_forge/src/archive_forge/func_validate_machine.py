import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def validate_machine(self, name_or_id, for_deploy=True):
    """Validate parameters of the machine.

        :param string name_or_id: The Name or UUID value representing the
            baremetal node.
        :param bool for_deploy: If ``True``, validate readiness for deployment,
            otherwise validate only the power management properties.
        :raises: :exc:`~openstack.exceptions.ValidationException`
        """
    if for_deploy:
        ifaces = ('boot', 'deploy', 'management', 'power')
    else:
        ifaces = ('power',)
    self.baremetal.validate_node(name_or_id, required=ifaces)