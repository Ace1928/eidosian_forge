import testtools
from openstack import exceptions
from openstack.tests.functional import base
def test_remove_volume_type_access_missing_project(self):
    with testtools.ExpectedException(exceptions.NotFoundException, 'Unable to revoke.*'):
        self.operator_cloud.remove_volume_type_access('test-volume-type', '00000000000000000000000000000000')