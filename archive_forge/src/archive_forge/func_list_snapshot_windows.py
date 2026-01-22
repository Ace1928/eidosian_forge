import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def list_snapshot_windows(self, location, plan):
    """
        List snapshot windows in a given location
        :param location: a location object or location id such as "NA9"
        :param plan: 'ESSENTIALS' or 'ADVANCED'
        :return: dictionary with keys id, day_of_week, start_hour, availability
        :rtype: dict
        """
    params = {}
    params['datacenterId'] = self._location_to_location_id(location)
    params['servicePlan'] = plan
    return self._to_windows(self.connection.request_with_orgId_api_2('infrastructure/snapshotWindow', params=params).object)