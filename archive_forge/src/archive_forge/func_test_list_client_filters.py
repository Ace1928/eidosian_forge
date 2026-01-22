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
def test_list_client_filters(self):
    qp = 'query param!'
    uri_param = 'uri param!'
    mock_empty = mock.Mock()
    mock_empty.status_code = 200
    mock_empty.links = {}
    mock_empty.json.return_value = {'resources': [{'a': '1', 'b': '1'}, {'a': '1', 'b': '2'}]}
    self.session.get.side_effect = [mock_empty]

    class Test(self.test_class):
        _query_mapping = resource.QueryParameters('a')
        base_path = '/%(something)s/blah'
        something = resource.URI('something')
        a = resource.Body('a')
        b = resource.Body('b')
    res = list(Test.list(self.session, paginated=True, query_param=qp, allow_unknown_params=True, something=uri_param, a='1', b='2'))
    self.session.get.assert_called_once_with('/{something}/blah'.format(something=uri_param), headers={'Accept': 'application/json'}, microversion=None, params={'a': '1'})
    self.assertEqual(1, len(res))
    self.assertEqual('2', res[0].b)