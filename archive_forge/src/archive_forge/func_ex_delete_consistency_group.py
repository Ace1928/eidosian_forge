import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_consistency_group(self, consistency_group_id):
    """
        Delete's a Consistency Group

        :param consistency_group_id: Id of Consistency Group to delete
        :type ``str``
        :return: True if response_code contains either
        IN_PROGRESS' or 'OK' otherwise False
        :rtype: ``bool``
        """
    delete_elm = ET.Element('deleteConsistencyGroup', {'id': consistency_group_id, 'xmlns': TYPES_URN})
    response = self.connection.request_with_orgId_api_2('consistencyGroup/deleteConsistencyGroup', method='POST', data=ET.tostring(delete_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']