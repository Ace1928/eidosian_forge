from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_list_flavors_undetailed(self):
    fl = self.cs.flavors.list(detailed=False)
    self.cs.assert_called('GET', '/flavors')
    for flavor in fl:
        self.assertTrue(hasattr(flavor, 'description'), '%s does not have a description set.' % flavor)