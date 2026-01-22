import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import volume
from openstack import exceptions
from openstack.tests.unit import base
def test__prepare_request_body(self):
    sot = volume.Volume(**VOLUME)
    body = sot._prepare_request_body(patch=False, prepend_key=True)
    original_body = copy.deepcopy(sot._body.dirty)
    self.assertEqual(original_body['OS-SCH-HNT:scheduler_hints'], body['OS-SCH-HNT:scheduler_hints'])
    original_body.pop('OS-SCH-HNT:scheduler_hints')
    self.assertEqual(original_body, body['volume'])