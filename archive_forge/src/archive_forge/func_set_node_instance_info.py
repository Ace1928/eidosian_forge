import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def set_node_instance_info(self, uuid, patch):
    warnings.warn('The set_node_instance_info call is deprecated, use patch_machine or update_machine instead', os_warnings.OpenStackDeprecationWarning)
    return self.patch_machine(uuid, patch)