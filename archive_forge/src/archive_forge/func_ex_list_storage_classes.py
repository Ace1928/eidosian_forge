import json
import time
import hashlib
from datetime import datetime
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.kubernetes import (
def ex_list_storage_classes(self):
    sc_req = '/apis/storage.k8s.io/v1/storageclasses'
    try:
        result = self.connection.request(sc_req).object
    except Exception:
        raise
    scs = [item['metadata']['name'] for item in result['items']]
    return scs