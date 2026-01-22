import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_invalid_adapter_opts(self):
    """Adapter opts are bogus, in exception-raising ways."""
    self.oslo_config_dict['heat'] = {'interface': 'public', 'valid_interfaces': 'private'}
    self.assert_service_disabled('orchestration', "Encountered an exception attempting to process config for project 'heat' (service type 'orchestration'): interface and valid_interfaces are mutually exclusive.")