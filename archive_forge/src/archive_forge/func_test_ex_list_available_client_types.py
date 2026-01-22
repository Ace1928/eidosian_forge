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
def test_ex_list_available_client_types(self):
    target = self.driver.list_targets()[0]
    answer = self.driver.ex_list_available_client_types(target)
    self.assertEqual(len(answer), 1)
    self.assertEqual(answer[0].type, 'FA.Linux')
    self.assertEqual(answer[0].is_file_system, True)
    self.assertEqual(answer[0].description, 'Linux File system')