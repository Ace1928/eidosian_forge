import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
@get_params
def ex_list_consistency_groups(self, params={}):
    """
        Functions takes a named parameter that must be one of the following
        :param params: A dictionary composed of one of the following keys
        and a value
        * target_data_center_id=
        * source_network_domain_id=
        * target_network_domain_id=
        * source_server_id=
        * target_server_id=
        * name=
        * state=
        * operation_status=
        * drs_infrastructure_status=
        :rtype:  `list` of :class: `NttCisConsistencyGroup`
        """
    response = self.connection.request_with_orgId_api_2('consistencyGroup/consistencyGroup', params=params).object
    cgs = self._to_consistency_groups(response)
    return cgs