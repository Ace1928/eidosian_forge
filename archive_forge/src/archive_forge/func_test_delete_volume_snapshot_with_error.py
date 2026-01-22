from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_volume_snapshot_with_error(self):
    """
        Test that a exception while deleting a volume snapshot will cause an
        SDKException.
        """
    fake_snapshot = fakes.FakeVolumeSnapshot('1234', 'available', 'foo', 'derpysnapshot')
    fake_snapshot_dict = meta.obj_to_munch(fake_snapshot)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', 'detail']), json={'snapshots': [fake_snapshot_dict]}), dict(method='DELETE', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', fake_snapshot_dict['id']]), status_code=404)])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_volume_snapshot, name_or_id='1234')
    self.assert_calls()