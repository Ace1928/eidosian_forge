import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_such_conf_section(self):
    """No conf section (therefore no adapter opts) for service type."""
    del self.oslo_config_dict['heat']
    self.assert_service_disabled('orchestration', "No section for project 'heat' (service type 'orchestration') was present in the config.")