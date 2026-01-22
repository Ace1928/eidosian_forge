import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_set_microversion_headers(self):
    headers = {}
    client_session.Session._set_microversion_headers(headers, '2.30', 'compute', None)
    self.assertEqual(headers['OpenStack-API-Version'], 'compute 2.30')
    self.assertEqual(headers['X-OpenStack-Nova-API-Version'], '2.30')
    self.assertEqual(len(headers.keys()), 2)
    headers = {}
    client_session.Session._set_microversion_headers(headers, (2, 30), None, {'service_type': 'compute'})
    self.assertEqual(headers['OpenStack-API-Version'], 'compute 2.30')
    self.assertEqual(headers['X-OpenStack-Nova-API-Version'], '2.30')
    self.assertEqual(len(headers.keys()), 2)
    headers = {}
    client_session.Session._set_microversion_headers(headers, 'latest', 'compute', None)
    self.assertEqual(headers['OpenStack-API-Version'], 'compute latest')
    self.assertEqual(headers['X-OpenStack-Nova-API-Version'], 'latest')
    self.assertEqual(len(headers.keys()), 2)
    headers = {}
    client_session.Session._set_microversion_headers(headers, (discover.LATEST, discover.LATEST), 'compute', None)
    self.assertEqual(headers['OpenStack-API-Version'], 'compute latest')
    self.assertEqual(headers['X-OpenStack-Nova-API-Version'], 'latest')
    self.assertEqual(len(headers.keys()), 2)
    headers = {}
    client_session.Session._set_microversion_headers(headers, '2.30', 'baremetal', None)
    self.assertEqual(headers['OpenStack-API-Version'], 'baremetal 2.30')
    self.assertEqual(headers['X-OpenStack-Ironic-API-Version'], '2.30')
    self.assertEqual(len(headers.keys()), 2)
    headers = {}
    client_session.Session._set_microversion_headers(headers, (2, 30), None, {'service_type': 'volumev2'})
    self.assertEqual(headers['OpenStack-API-Version'], 'volume 2.30')
    self.assertEqual(len(headers.keys()), 1)
    headers = {}
    client_session.Session._set_microversion_headers(headers, (2, 30), None, {'service_type': 'block-storage'})
    self.assertEqual(headers['OpenStack-API-Version'], 'volume 2.30')
    self.assertEqual(len(headers.keys()), 1)
    for service_type in ['sharev2', 'shared-file-system']:
        headers = {}
        client_session.Session._set_microversion_headers(headers, (2, 30), None, {'service_type': service_type})
        self.assertEqual(headers['X-OpenStack-Manila-API-Version'], '2.30')
        self.assertEqual(headers['OpenStack-API-Version'], 'shared-file-system 2.30')
        self.assertEqual(len(headers.keys()), 2)
    headers = {'OpenStack-API-Version': 'compute 2.30', 'X-OpenStack-Nova-API-Version': '2.30'}
    client_session.Session._set_microversion_headers(headers, (2, 31), None, {'service_type': 'volume'})
    self.assertEqual(headers['OpenStack-API-Version'], 'compute 2.30')
    self.assertEqual(headers['X-OpenStack-Nova-API-Version'], '2.30')
    self.assertRaises(TypeError, client_session.Session._set_microversion_headers, {}, '2.latest', 'service_type', None)
    self.assertRaises(TypeError, client_session.Session._set_microversion_headers, {}, (2, discover.LATEST), 'service_type', None)
    self.assertRaises(TypeError, client_session.Session._set_microversion_headers, {}, 'bogus', 'service_type', None)
    self.assertRaises(TypeError, client_session.Session._set_microversion_headers, {}, (2, 30), None, None)
    self.assertRaises(TypeError, client_session.Session._set_microversion_headers, {}, (2, 30), None, {'no_service_type': 'here'})