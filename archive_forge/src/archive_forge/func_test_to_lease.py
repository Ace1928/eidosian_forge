import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def test_to_lease(self):
    res = self.driver.connection.request(get_url_path('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6d'), headers={'Content-Type': 'application/vnd.vmware.vcloud.vApp+xml'})
    lease_settings_section = res.object.find(fixxpath(res.object, 'LeaseSettingsSection'))
    lease = Lease.to_lease(lease_element=lease_settings_section)
    self.assertEqual(lease.deployment_lease, 86400)
    self.assertEqual(lease.deployment_lease_expiration, datetime.datetime(year=2019, month=10, day=7, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC))
    self.assertEqual(lease.storage_lease, 172800)
    self.assertEqual(lease.storage_lease_expiration, datetime.datetime(year=2019, month=10, day=8, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC))