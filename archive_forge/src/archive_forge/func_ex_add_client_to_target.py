from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def ex_add_client_to_target(self, target, client_type, storage_policy, schedule_policy, trigger, email):
    """
        Add a client to a target

        :param target: Backup target with the backup data
        :type  target: Instance of :class:`BackupTarget` or ``str``

        :param client: Client to add to the target
        :type  client: Instance of :class:`DimensionDataBackupClientType`
                       or ``str``

        :param storage_policy: The storage policy for the client
        :type  storage_policy: Instance of
                               :class:`DimensionDataBackupStoragePolicy`
                               or ``str``

        :param schedule_policy: The schedule policy for the client
        :type  schedule_policy: Instance of
                                :class:`DimensionDataBackupSchedulePolicy`
                                or ``str``

        :param trigger: The notify trigger for the client
        :type  trigger: ``str``

        :param email: The notify email for the client
        :type  email: ``str``

        :rtype: ``bool``
        """
    server_id = self._target_to_target_address(target)
    backup_elm = ET.Element('NewBackupClient', {'xmlns': BACKUP_NS})
    if isinstance(client_type, DimensionDataBackupClientType):
        ET.SubElement(backup_elm, 'type').text = client_type.type
    else:
        ET.SubElement(backup_elm, 'type').text = client_type
    if isinstance(storage_policy, DimensionDataBackupStoragePolicy):
        ET.SubElement(backup_elm, 'storagePolicyName').text = storage_policy.name
    else:
        ET.SubElement(backup_elm, 'storagePolicyName').text = storage_policy
    if isinstance(schedule_policy, DimensionDataBackupSchedulePolicy):
        ET.SubElement(backup_elm, 'schedulePolicyName').text = schedule_policy.name
    else:
        ET.SubElement(backup_elm, 'schedulePolicyName').text = schedule_policy
    alerting_elm = ET.SubElement(backup_elm, 'alerting')
    alerting_elm.set('trigger', trigger)
    ET.SubElement(alerting_elm, 'emailAddress').text = email
    response = self.connection.request_with_orgId_api_1('server/%s/backup/client' % server_id, method='POST', data=ET.tostring(backup_elm)).object
    response_code = findtext(response, 'result', GENERAL_NS)
    return response_code in ['IN_PROGRESS', 'SUCCESS']