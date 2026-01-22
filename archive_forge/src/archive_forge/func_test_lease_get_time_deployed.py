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
def test_lease_get_time_deployed(self):
    deployment_datetime = datetime.datetime(year=2019, month=10, day=6, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC)
    deployment_lease_exp_actual = datetime.datetime(year=2019, month=10, day=7, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC)
    storage_lease_exp_actual = datetime.datetime(year=2019, month=10, day=8, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC)
    assertion_error = False
    for deployment_lease, storage_lease, deployment_lease_exp, storage_lease_exp, exception, res in ((None, None, None, None, True, None), (None, None, None, storage_lease_exp_actual, True, None), (None, None, deployment_lease_exp_actual, None, True, None), (None, None, deployment_lease_exp_actual, storage_lease_exp_actual, True, None), (None, 172800, None, None, True, None), (None, 172800, None, storage_lease_exp_actual, False, deployment_datetime), (None, 172800, deployment_lease_exp_actual, None, True, deployment_datetime), (None, 172800, deployment_lease_exp_actual, storage_lease_exp_actual, False, deployment_datetime), (86400, None, None, None, True, None), (86400, None, None, storage_lease_exp_actual, True, None), (86400, None, deployment_lease_exp_actual, None, False, deployment_datetime), (86400, None, deployment_lease_exp_actual, storage_lease_exp_actual, False, deployment_datetime), (86400, 172800, None, None, True, None), (86400, 172800, None, storage_lease_exp_actual, False, deployment_datetime), (86400, 172800, deployment_lease_exp_actual, None, False, deployment_datetime), (86400, 172800, deployment_lease_exp_actual, storage_lease_exp_actual, False, deployment_datetime)):
        try:
            lease = Lease('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6a/leaseSettingsSection/', deployment_lease=deployment_lease, storage_lease=storage_lease, deployment_lease_expiration=deployment_lease_exp, storage_lease_expiration=storage_lease_exp)
            if exception:
                with assertRaisesRegex(self, Exception, re.escape('Cannot get time deployed. Missing complete lease and expiration information.')):
                    lease.get_deployment_time()
            else:
                self.assertEqual(lease.get_deployment_time(), res)
        except AssertionError:
            assertion_error = True
            print_parameterized_failure([('deployment_lease', deployment_lease), ('storage_lease', storage_lease), ('deployment_lease_exp', deployment_lease_exp), ('storage_lease_exp', storage_lease_exp), ('exception', exception), ('res', res)])
    if assertion_error:
        self.fail(msg='Assertion error(s) encountered. Details above.')