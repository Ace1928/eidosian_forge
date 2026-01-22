import atexit
import concurrent.futures
import importlib.metadata as importlib_metadata
import warnings
import weakref
import keystoneauth1.exceptions
import requestsexceptions
from openstack import _log
from openstack import _services_mixin
from openstack.cloud import _accelerator
from openstack.cloud import _baremetal
from openstack.cloud import _block_storage
from openstack.cloud import _coe
from openstack.cloud import _compute
from openstack.cloud import _dns
from openstack.cloud import _floating_ip
from openstack.cloud import _identity
from openstack.cloud import _image
from openstack.cloud import _network
from openstack.cloud import _network_common
from openstack.cloud import _object_store
from openstack.cloud import _orchestration
from openstack.cloud import _security_group
from openstack.cloud import _shared_file_system
from openstack.cloud import openstackcloud as _cloud
from openstack import config as _config
from openstack.config import cloud_region
from openstack import exceptions
from openstack import service_description
def set_global_request_id(self, global_request_id):
    self._global_request_id = global_request_id