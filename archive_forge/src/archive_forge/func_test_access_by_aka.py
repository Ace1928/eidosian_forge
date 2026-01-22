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
def test_access_by_aka(self):

    class Test(resource.Resource):
        foo = resource.Header('foo_remote', aka='foo_alias')
    res = Test(foo='bar', name='test')
    self.assertEqual('bar', res['foo_alias'])
    self.assertEqual('bar', res.foo_alias)
    self.assertTrue('foo' in res.keys())
    self.assertTrue('foo_alias' in res.keys())
    expected = utils.Munch({'id': None, 'name': 'test', 'location': None, 'foo': 'bar', 'foo_alias': 'bar'})
    actual = utils.Munch(res)
    self.assertEqual(expected, actual)
    self.assertEqual(expected, res.toDict())
    self.assertEqual(expected, res.to_dict())
    self.assertDictEqual(expected, res)
    self.assertDictEqual(expected, dict(res))