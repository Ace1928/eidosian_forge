from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def list_targets(self):
    """
        List all backuptargets

        :rtype: ``list`` of :class:`BackupTarget`
        """
    targets = self._to_targets(self.connection.request_with_orgId_api_2('server/server').object)
    return targets