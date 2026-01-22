import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_tag_key(self, name, description=None, value_required=True, display_on_report=True):
    """
        Creates a tag key in the NTTC-CIS Cloud

        :param name: The name of the tag key (required)
        :type  name: ``str``

        :param description: The description of the tag key
        :type  description: ``str``

        :param value_required: If a value is required for the tag
                               Tags themselves can be just a tag,
                               or be a key/value pair
        :type  value_required: ``bool``

        :param display_on_report: Should this key show up on the usage reports
        :type  display_on_report: ``bool``

        :rtype: ``bool``
        """
    create_tag_key = ET.Element('createTagKey', {'xmlns': TYPES_URN})
    ET.SubElement(create_tag_key, 'name').text = name
    if description is not None:
        ET.SubElement(create_tag_key, 'description').text = description
    ET.SubElement(create_tag_key, 'valueRequired').text = str(value_required).lower()
    ET.SubElement(create_tag_key, 'displayOnReport').text = str(display_on_report).lower()
    response = self.connection.request_with_orgId_api_2('tag/createTagKey', method='POST', data=ET.tostring(create_tag_key)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']