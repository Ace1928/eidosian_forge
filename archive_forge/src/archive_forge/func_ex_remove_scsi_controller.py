import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_remove_scsi_controller(self, controller_id):
    """
        Added 8/27/18:  Adds a SCSI Controller by node id
        :param controller_id: Scsi controller's id
        :return: whether addition is in progress or 'OK' otherwise false
        """
    update_node = ET.Element('removeScsiController', {'xmlns': TYPES_URN})
    update_node.set('id', controller_id)
    result = self.connection.request_with_orgId_api_2('server/removeScsiController', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']