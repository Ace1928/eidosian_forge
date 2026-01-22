from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def suspend_target_job(self, target, job):
    """
        Suspend a running backup job on a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget`

        :param job: Backup target job to suspend
        :type  job: Instance of :class:`BackupTargetJob`

        :rtype: ``bool``
        """
    raise NotImplementedError('suspend_target_job not implemented for this driver')