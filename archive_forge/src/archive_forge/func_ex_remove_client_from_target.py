from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def ex_remove_client_from_target(self, target, backup_client):
    """
        Removes a client from a backup target

        :param  target: The backup target to remove the client from
        :type   target: :class:`BackupTarget` or ``str``

        :param  backup_client: The backup client to remove
        :type   backup_client: :class:`DimensionDataBackupClient` or ``str``

        :rtype: ``bool``
        """
    server_id = self._target_to_target_address(target)
    client_id = self._client_to_client_id(backup_client)
    response = self.connection.request_with_orgId_api_1('server/{}/backup/client/{}?disable'.format(server_id, client_id), method='GET').object
    response_code = findtext(response, 'result', GENERAL_NS)
    return response_code in ['IN_PROGRESS', 'SUCCESS']