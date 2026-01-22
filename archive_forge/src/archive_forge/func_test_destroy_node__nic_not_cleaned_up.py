import sys
import json
import functools
from datetime import datetime
from unittest import mock
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib, parse_qs, urlparse, urlunquote
from libcloud.common.types import LibcloudError
from libcloud.compute.base import NodeSize, NodeLocation, StorageVolume, VolumeSnapshot
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.azure_arm import (
@mock.patch('time.sleep', return_value=None)
def test_destroy_node__nic_not_cleaned_up(self, time_sleep_mock):

    def error(e, **kwargs):
        raise e(**kwargs)
    node = self.driver.list_nodes()[0]
    AzureMockHttp.responses = [lambda f: (httplib.OK, None, {}, 'OK'), lambda f: error(BaseHTTPError, code=404, message='Not found'), lambda f: error(BaseHTTPError, code=500, message='Cloud weather')]
    with self.assertRaises(BaseHTTPError):
        self.driver.destroy_node(node)