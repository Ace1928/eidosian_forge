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
def test_ex_add_client_to_target_STR(self):
    self.assertTrue(self.driver.ex_add_client_to_target('e75ead52-692f-4314-8725-c8a4f4d13a87', 'FA.Linux', '14 Day Storage Policy', '12AM - 6AM', 'ON_FAILURE', 'nobody@example.com'))