import warnings
import testtools
from openstack import exceptions
from openstack import proxy
from openstack.tests.unit import base
def test_unsupported_version(self):
    with testtools.ExpectedException(exceptions.NotSupported):
        self.cloud.image.get('/')
    self.assert_calls()