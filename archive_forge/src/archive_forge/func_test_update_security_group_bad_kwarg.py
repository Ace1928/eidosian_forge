import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_security_group_bad_kwarg(self):
    self.assertRaises(TypeError, self.cloud.update_security_group, 'doesNotExist', bad_arg='')