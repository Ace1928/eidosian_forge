import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_base_image_by_id(self, id):
    """
        Gets a Base image in the NTTC-CIS Cloud given the id

        :param id: The id of the image
        :type  id: ``str``

        :rtype: :class:`NodeImage`
        """
    image = self.connection.request_with_orgId_api_2('image/osImage/%s' % id).object
    return self._to_image(image)