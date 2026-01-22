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
def test__update(self):
    sot = resource.Resource()
    body = 'body'
    header = 'header'
    uri = 'uri'
    computed = 'computed'
    sot._collect_attrs = mock.Mock(return_value=(body, header, uri, computed))
    sot._body.update = mock.Mock()
    sot._header.update = mock.Mock()
    sot._uri.update = mock.Mock()
    sot._computed.update = mock.Mock()
    args = {'arg': 1}
    sot._update(**args)
    sot._collect_attrs.assert_called_once_with(args)
    sot._body.update.assert_called_once_with(body)
    sot._header.update.assert_called_once_with(header)
    sot._uri.update.assert_called_once_with(uri)
    sot._computed.update.assert_called_with(computed)