import testtools
from openstack import exceptions
from openstack.tests.functional import base
def test_add_volume_type_access_missing_volume(self):
    with testtools.ExpectedException(exceptions.SDKException, 'VolumeType not found.*'):
        self.operator_cloud.add_volume_type_access('MISSING_VOLUME_TYPE', self.operator_cloud.current_project_id)