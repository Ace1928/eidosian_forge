import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_consistency_group(self, name, journal_size_gb, source_server_id, target_server_id, description=None):
    """
        Create a consistency group

        :param name: Name of consistency group
        :type name: ``str``

        :param journal_size_gb: Journal size in GB
        :type journal_size_gb: ``str``

        :param source_server_id: Id of the server to copy
        :type source_server_id: ``str``

        :param target_server_id: Id of the server to receive the copy
        :type: target_server_id: ``str``

        :param description: (Optional) Description of consistency group
        :type: description: ``str``

        :rtype: :class:`NttCisConsistencyGroup`
        """
    consistency_group_elm = ET.Element('createConsistencyGroup', {'xmlns': TYPES_URN})
    ET.SubElement(consistency_group_elm, 'name').text = name
    if description is not None:
        ET.SubElement(consistency_group_elm, 'description').text = description
    ET.SubElement(consistency_group_elm, 'journalSizeGb').text = journal_size_gb
    server_pair = ET.SubElement(consistency_group_elm, 'serverPair')
    ET.SubElement(server_pair, 'sourceServerId').text = source_server_id
    ET.SubElement(server_pair, 'targetServerId').text = target_server_id
    response = self.connection.request_with_orgId_api_2('consistencyGroup/createConsistencyGroup', method='POST', data=ET.tostring(consistency_group_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']