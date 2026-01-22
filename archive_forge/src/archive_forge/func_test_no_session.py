import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_no_session(self):
    self.assertRaises(exceptions.ConfigException, cloud_region.from_conf, self._load_ks_cfg_opts())