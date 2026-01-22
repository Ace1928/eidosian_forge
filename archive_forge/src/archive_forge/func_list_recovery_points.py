from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def list_recovery_points(self, target, start_date=None, end_date=None):
    """
        List the recovery points available for a target

        :param target: Backup target to delete
        :type  target: Instance of :class:`BackupTarget`

        :param start_date: The start date to show jobs between (optional)
        :type  start_date: :class:`datetime.datetime`

        :param end_date: The end date to show jobs between (optional)
        :type  end_date: :class:`datetime.datetime``

        :rtype: ``list`` of :class:`BackupTargetRecoveryPoint`
        """
    raise NotImplementedError('list_recovery_points not implemented for this driver')