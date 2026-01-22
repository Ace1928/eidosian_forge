import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_get_all_regions_with_renames(self):
    rename_map = {'ec3': 'ec2'}
    resolver = BotoEndpointResolver(endpoint_data=self._endpoint_data(), service_rename_map=rename_map)
    regions = sorted(resolver.get_all_available_regions('ec2'))
    expected_regions = sorted(['us-bar', 'eu-baz', 'us-foo', 'foo-1', 'foo-2', 'foo-3'])
    self.assertEqual(regions, expected_regions)