from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
@mock.patch('glance.location._check_image_location', new=lambda *a: 0)
@mock.patch('glance.location.ImageRepoProxy._set_acls', new=lambda *a: 0)
def test_image_update_locations(self):
    self.config(show_multiple_locations=True)
    self.start_server()
    image_id = self._create_and_upload()
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://foo.bar', 'metadata': {}}})
    self.assertEqual(200, resp.status_code, resp.text)
    self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
    self.assertEqual('http://foo.bar', self.api_get('/v2/images/%s' % image_id).json['locations'][1]['url'])
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/locations/0'})
    self.assertEqual(200, resp.status_code, resp.text)
    self.assertEqual(1, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://foo.baz', 'metadata': {}}})
    self.assertEqual(200, resp.status_code, resp.text)
    self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
    self.set_policy_rules({'get_image': '', 'get_image_location': '', 'set_image_location': '!', 'delete_image_location': '!'})
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/locations/0'})
    self.assertEqual(403, resp.status_code, resp.text)
    self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/locations/0', 'value': {'url': 'http://foo.baz', 'metadata': {}}})
    self.assertEqual(403, resp.status_code, resp.text)
    self.assertEqual(2, len(self.api_get('/v2/images/%s' % image_id).json['locations']))