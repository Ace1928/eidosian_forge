from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def get_supported_target_types(self):
    """
        Get a list of backup target types this driver supports

        :return: ``list`` of :class:``BackupTargetType``
        """
    return [BackupTargetType.VIRTUAL]