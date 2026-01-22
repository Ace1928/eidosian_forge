import ddt
from manilaclient.tests.functional import base
def test_set_and_get_metadata(self):
    share = self.create_share(cleanup_in_class=False, client=self.get_user_client())
    md = {'key3': 'value3', 'key4': 'value4'}
    self.user_client.set_share_metadata(share['id'], md)
    metadata = self.user_client.get_share_metadata(share['id'])
    self.assertEqual(2, len(metadata))
    self.assertIn('key3', metadata)
    self.assertIn('key4', metadata)
    self.assertEqual(md['key3'], metadata['key3'])
    self.assertEqual(md['key4'], metadata['key4'])