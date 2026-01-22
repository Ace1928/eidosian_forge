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
def test_destroy_node__destroy_nic_retries(self, time_sleep_mock):

    def error(e, **kwargs):
        raise e(**kwargs)
    node = self.driver.list_nodes()[0]
    err = BaseHTTPError(code=400, message='[NicInUse] Cannot destroy')
    with mock.patch.object(self.driver, 'ex_destroy_nic') as m:
        m.side_effect = [err] * 5 + [True]
        ret = self.driver.destroy_node(node)
        self.assertTrue(ret)
        self.assertEqual(6, m.call_count)
        m.side_effect = [err] * 10 + [True]
        with self.assertRaises(BaseHTTPError):
            self.driver.destroy_node(node)
            self.assertEqual(10, m.call_count)