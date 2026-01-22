import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_consistency_group_snapshots(self, consistency_group_id, create_time_min=None, create_time_max=None):
    """
        Optional parameters identify the date of creation of Consistency Group
        snapshots in *XML Schema (XSD) date time format. Best used as a
        combination of createTime.MIN and createTime.MAX. If neither is
        provided then all snapshots up to the possible maximum of 1014
        will be returned. If MIN is provided by itself, all snapshots
        between the time specified by MIN and the point in time of
        execution will be returned. If MAX is provided by itself,
        then all snapshots up to that point in time (up to the
        maximum number of 1014) will be returned. MIN and MAX are
        inclusive for this API function

        :param consistency_group_id: The id of consistency group
        :type consistency_group_id: ``str``

        :param create_time_min: (Optional) in form YYYY-MM-DDT00:00.00.00Z or
                                           substitute time offset for Z, i.e,
                                           -05:00
        :type create_time_min: ``str``

        :param create_time_max: (Optional) in form YYYY-MM-DDT00:00:00.000Z or
                                           substitute time offset for Z, i.e,
                                           -05:00
        :type create_time_max: ``str``

        :rtype: `list` of :class:`NttCisSnapshots`
        """
    if create_time_min is None and create_time_max is None:
        params = {'consistencyGroupId': consistency_group_id}
    elif create_time_min and create_time_max:
        params = {'consistencyGroupId': consistency_group_id, 'createTime.MIN': create_time_min, 'createTime.MAX': create_time_max}
    elif create_time_min or create_time_max:
        if create_time_max is not None:
            params = {'consistencyGroupId': consistency_group_id, 'createTime.MAX': create_time_max}
        elif create_time_min is not None:
            params = {'consistencyGroupId': consistency_group_id, 'createTime.MIN': create_time_min}
    paged_result = self.connection.request_with_orgId_api_2('consistencyGroup/snapshot', method='GET', params=params).object
    snapshots = self._to_process(paged_result)
    return snapshots