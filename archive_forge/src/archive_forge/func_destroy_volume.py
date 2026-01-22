import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def destroy_volume(self, volume):
    self._api_request('/disks/delete', {'diskId': int(volume.id)})
    return True