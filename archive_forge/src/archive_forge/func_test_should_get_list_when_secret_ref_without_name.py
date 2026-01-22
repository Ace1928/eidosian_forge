from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_list_when_secret_ref_without_name(self):
    container_resp = self.container.get_dict(self.entity_href)
    del container_resp.get('secret_refs')[0]['name']
    data = {'containers': [container_resp for v in range(3)]}
    self.responses.get(self.entity_base, json=data)
    containers_list = self.manager.list(limit=10, offset=5)
    self.assertTrue(len(containers_list) == 3)
    self.assertIsInstance(containers_list[0], containers.Container)
    self.assertEqual(self.entity_href, containers_list[0].container_ref)
    self.assertEqual(self.entity_base, self.responses.last_request.url.split('?')[0])
    for container in containers_list:
        for name in container._secret_refs.keys():
            self.assertIsNone(name)