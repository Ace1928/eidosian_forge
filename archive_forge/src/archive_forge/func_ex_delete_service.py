import json
import time
import hashlib
from datetime import datetime
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.kubernetes import (
def ex_delete_service(self, namespace, service_name):
    req = '{}/namespaces/{}/services/{}'.format(ROOT_URL, namespace, service_name)
    headers = {'Content-Type': 'application/yaml'}
    try:
        result = self.connection.request(req, method='DELETE', headers=headers)
    except Exception:
        raise
    return result.status in VALID_RESPONSE_CODES