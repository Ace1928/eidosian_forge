import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def deactivate_node(self, uuid, wait=False, timeout=1200):
    self.node_set_provision_state(uuid, 'deleted', wait=wait, timeout=timeout)