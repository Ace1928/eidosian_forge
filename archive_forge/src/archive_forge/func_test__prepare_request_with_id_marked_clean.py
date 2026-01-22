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
def test__prepare_request_with_id_marked_clean(self):

    class Test(resource.Resource):
        base_path = '/something'
        body_attr = resource.Body('x')
        header_attr = resource.Header('y')
    the_id = 'id'
    body_value = 'body'
    header_value = 'header'
    sot = Test(id=the_id, body_attr=body_value, header_attr=header_value, _synchronized=False)
    sot._body._dirty.discard('id')
    result = sot._prepare_request(requires_id=True)
    self.assertEqual('something/id', result.url)
    self.assertEqual({'x': body_value}, result.body)
    self.assertEqual({'y': header_value}, result.headers)