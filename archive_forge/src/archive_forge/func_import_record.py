from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
def import_record(self, backup_service, backup_url):
    """Import volume backup metadata record.

        :param backup_service: Backup service to use for importing the backup
        :param backup_url: Backup URL for importing the backup metadata
        :rtype: A dictionary containing volume backup metadata.
        """
    body = {'backup-record': {'backup_service': backup_service, 'backup_url': backup_url}}
    self.run_hooks('modify_body_for_update', body, 'backup-record')
    resp, body = self.api.client.post('/backups/import_record', body=body)
    return common_base.DictWithMeta(body['backup'], resp)