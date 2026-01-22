from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.backup.types import BackupTargetType
def recover_to(self, recovery_target, path=None):
    """
        Recover this recovery point out of place

        :param recovery_target: Backup target with to recover the data to
        :type  recovery_target: Instance of :class:`.BackupTarget`

        :param path: The part of the recovery point to recover (optional)
        :type  path: ``str``

        :rtype: Instance of :class:`.BackupTargetJob`
        """
    return self.driver.recover_target_out_of_place(target=self.target, recovery_point=self, recovery_target=recovery_target, path=path)