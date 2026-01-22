from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.common.google import GoogleResponse, GoogleBaseConnection
from libcloud.utils.iso8601 import parse_date
def ex_get_snapshot(self, name):
    request = '/global/snapshots/%s' % name
    response = self.connection.request(request, method='GET').object
    return response