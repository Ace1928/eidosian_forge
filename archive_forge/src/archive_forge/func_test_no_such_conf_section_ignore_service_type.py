import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_such_conf_section_ignore_service_type(self):
    """Ignore absent conf section if service type not requested."""
    del self.oslo_config_dict['heat']
    self.assert_service_disabled('orchestration', 'Not in the list of requested service_types.', service_types=['compute'])