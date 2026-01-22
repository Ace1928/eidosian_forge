import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def list_locations(self):
    """
        List data centers available.

        :return: list of node location objects
        :rtype: ``list`` of :class:`.NodeLocation`
        """
    return [NodeLocation(driver=self, **copy.deepcopy(location)) for location in SCALEWAY_LOCATION_DATA]