import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_migrate_preview_server(self, preview_id):
    migrate_preview = ET.Element('migrateSnapshotPreviewServer', {'xmlns': TYPES_URN, 'serverId': preview_id})
    result = self.connection.request_with_orgId_api_2('snapshot/migrateSnapshotPreviewServer', method='POST', data=ET.tostring(migrate_preview)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']