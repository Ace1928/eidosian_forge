from cinderclient.tests.functional import base
def test_snapshot_create_metadata(self):
    """Test steps:

        1) create volume in Setup()
        2) create snapshot with metadata
        3) check that metadata complies entered
        """
    snapshot = self.object_create('snapshot', params='--metadata test_metadata=test_date {0}'.format(self.volume['id']))
    self.assertEqual(str({'test_metadata': 'test_date'}), snapshot['metadata'])
    self.object_delete('snapshot', snapshot['id'])
    self.check_object_deleted('snapshot', snapshot['id'])