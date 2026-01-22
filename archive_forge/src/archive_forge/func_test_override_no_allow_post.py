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
def test_override_no_allow_post(self):
    attr_info = {'key': {'allow_post': False, 'default': constants.ATTR_NOT_SPECIFIED}}
    attr_inst = attributes.AttributeInfo(attr_info)
    self._test_fill_default_value(attr_inst, {'key': 'X'}, {'key': 'X'}, check_allow_post=False)