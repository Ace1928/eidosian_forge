import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def ex_list_zones(self, region_id=None):
    """
        List availability zones in the given region or the current region.

        :keyword region_id: the id of the region to query zones from
        :type region_id: ``str``

        :return: list of zones
        :rtype: ``list`` of ``ECSZone``
        """
    params = {'Action': 'DescribeZones'}
    if region_id:
        params['RegionId'] = region_id
    else:
        params['RegionId'] = self.region
    resp_body = self.connection.request(self.path, params).object
    zone_elements = findall(resp_body, 'Zones/Zone', namespace=self.namespace)
    zones = [self._to_zone(el) for el in zone_elements]
    return zones