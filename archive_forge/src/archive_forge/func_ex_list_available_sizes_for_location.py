import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
def ex_list_available_sizes_for_location(self, location: NodeLocation) -> List[str]:
    """Get a list of available sizes for the given location.

        :param location: The location to get available sizes for.
        :type location: :class: `NodeLocation`

        :return:  A list of available size IDs for the given location.
        :rtype: ``list`` of ``str``
        """
    resp = self.connection.request('/v2/regions/%s/availability' % location.id)
    return resp.object['available_plans']