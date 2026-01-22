import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_resolve_hostname(self):
    resolver = BotoEndpointResolver(self._endpoint_data())
    hostname = resolver.resolve_hostname('ec2', 'us-foo')
    expected_hostname = 'ec2.us-foo.amazonaws.com'
    self.assertEqual(hostname, expected_hostname)