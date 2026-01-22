from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_import_glance_direct(self):
    self.start_server()
    image_id = self._create_and_stage(visibility='public')
    self.set_policy_rules({'get_image': '', 'communitize_image': '', 'add_image': '', 'modify_image': ''})
    store_to_import = ['store1']
    response = self._import_direct(image_id, store_to_import)
    self.assertEqual(202, response.status_code)
    self._wait_for_import(image_id)
    self.assertEqual('success', self._get_latest_task(image_id)['status'])
    image_id = self._create_and_stage(visibility='community')
    headers = self._headers({'X-Roles': 'member'})
    response = self._import_direct(image_id, store_to_import, headers=headers)
    self.assertEqual(202, response.status_code)
    self._wait_for_import(image_id)
    self.assertEqual('success', self._get_latest_task(image_id)['status'])
    image_id = self._create_and_stage(visibility='community')
    self.set_policy_rules({'get_image': '', 'modify_image': '!'})
    headers = self._headers({'X-Roles': 'member', 'X-Project-Id': 'fake-project-id'})
    response = self._import_direct(image_id, store_to_import, headers=headers)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
    headers = self._headers({'X-Roles': 'member', 'X-Project-Id': 'fake-project-id'})
    response = self._import_direct(image_id, store_to_import, headers=headers)
    self.assertEqual(404, response.status_code)