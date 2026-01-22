from cinderclient.tests.functional import base
def test_volume_create_delete_id(self):
    """Create and delete a volume by ID."""
    volume = self.object_create('volume', params='1')
    self.assert_object_details(self.CREATE_VOLUME_PROPERTY, volume.keys())
    self.object_delete('volume', volume['id'])
    self.check_object_deleted('volume', volume['id'])