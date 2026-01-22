from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
def export_record(self, backup_id):
    """Export volume backup metadata record.

        :param backup_id: The ID of the backup to export.
        :rtype: A dictionary containing 'backup_url' and 'backup_service'.
        """
    resp, body = self.api.client.get('/backups/%s/export_record' % backup_id)
    return common_base.DictWithMeta(body['backup-record'], resp)