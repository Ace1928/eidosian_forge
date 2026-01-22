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
def ex_destroy_cloud_service(self, name):
    """
        Delete an azure cloud service.

        :param      name: Name of the cloud service to destroy.
        :type       name: ``str``

        :rtype: ``bool``
        """
    response = self._perform_cloud_service_delete(self._get_hosted_service_path(name))
    self.raise_for_response(response, 200)
    return True