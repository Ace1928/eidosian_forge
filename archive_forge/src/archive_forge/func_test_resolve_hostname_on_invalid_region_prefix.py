import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_resolve_hostname_on_invalid_region_prefix(self):
    resolver = BotoEndpointResolver(self._endpoint_data())
    hostname = resolver.resolve_hostname('s3', 'fake-west-1')
    self.assertIsNone(hostname)