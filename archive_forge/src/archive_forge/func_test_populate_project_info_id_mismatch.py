from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
def test_populate_project_info_id_mismatch(self):
    attrs = {'project_id': uuidutils.generate_uuid(), 'tenant_id': uuidutils.generate_uuid()}
    self.assertRaises(exc.HTTPBadRequest, attributes.populate_project_info, attrs)