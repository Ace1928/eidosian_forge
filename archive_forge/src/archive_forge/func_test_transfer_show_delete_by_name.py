from cinderclient.tests.functional import base
def test_transfer_show_delete_by_name(self):
    """Show volume transfer by name"""
    volume = self.object_create('volume', params='1')
    self.object_create('transfer', params='%s --name TEST_TRANSFER_SHOW' % volume['id'])
    output = self.cinder('transfer-show', params='TEST_TRANSFER_SHOW')
    transfer = self._get_property_from_output(output)
    self.assertEqual('TEST_TRANSFER_SHOW', transfer['name'])
    self.assert_object_details(self.TRANSFER_SHOW_PROPERTY, transfer.keys())
    self.object_delete('transfer', 'TEST_TRANSFER_SHOW')
    self.check_object_deleted('transfer', 'TEST_TRANSFER_SHOW')
    self.object_delete('volume', volume['id'])
    self.check_object_deleted('volume', volume['id'])