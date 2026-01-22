import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.backup.base import BackupTargetJob
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import BackupFileFixtures
from libcloud.common.dimensiondata import DimensionDataAPIException
from libcloud.backup.drivers.dimensiondata import DEFAULT_BACKUP_PLAN
from libcloud.backup.drivers.dimensiondata import DimensionDataBackupDriver as DimensionData
def test_ex_add_client_to_target(self):
    target = self.driver.list_targets()[0]
    client = self.driver.ex_list_available_client_types(target)[0]
    storage_policy = self.driver.ex_list_available_storage_policies(target)[0]
    schedule_policy = self.driver.ex_list_available_schedule_policies(target)[0]
    self.assertTrue(self.driver.ex_add_client_to_target(target, client, storage_policy, schedule_policy, 'ON_FAILURE', 'nobody@example.com'))