import json
import time
import hashlib
from datetime import datetime
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.kubernetes import (
def ex_list_persistent_volume_claims(self, namespace='default'):
    pvc_req = ROOT_URL + 'namespaces/' + namespace + '/persistentvolumeclaims'
    try:
        result = self.connection.request(pvc_req).object
    except Exception:
        raise
    pvcs = [item['metadata']['name'] for item in result['items']]
    return pvcs