import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_patch_with_sdk_names(self):

    class Test(resource.Resource):
        allow_patch = True
        id = resource.Body('id')
        attr = resource.Body('attr')
        nested = resource.Body('renamed')
        other = resource.Body('other')
    test_patch = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/nested/dog', 'op': 'remove'}, {'path': '/nested/cat', 'op': 'add', 'value': 'meow'}]
    expected = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/renamed/dog', 'op': 'remove'}, {'path': '/renamed/cat', 'op': 'add', 'value': 'meow'}]
    sot = Test.existing(id=1, attr=42, nested={'dog': 'bark'})
    sot.patch(self.session, test_patch)
    self.session.patch.assert_called_once_with('/1', json=expected, headers=mock.ANY, microversion=None)