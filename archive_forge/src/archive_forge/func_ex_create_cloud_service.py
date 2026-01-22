import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def ex_create_cloud_service(self, name, location, description=None, extended_properties=None):
    """
        Create an azure cloud service.

        :param      name: Name of the service to create
        :type       name: ``str``

        :param      location: Standard azure location string
        :type       location: ``str``

        :param      description: Optional description
        :type       description: ``str``

        :param      extended_properties: Optional extended_properties
        :type       extended_properties: ``dict``

        :rtype: ``bool``
        """
    response = self._perform_cloud_service_create(self._get_hosted_service_path(), AzureXmlSerializer.create_hosted_service_to_xml(name, self._encode_base64(name), description, location, None, extended_properties))
    self.raise_for_response(response, 201)
    return True