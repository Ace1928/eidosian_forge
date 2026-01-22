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
@staticmethod
def doc_from_xml(document_element_name, inner_xml=None):
    """
        Wraps the specified xml in an xml root element with default azure
        namespaces
        """
    '\n        nsmap = {\n            None: "http://www.w3.org/2001/XMLSchema-instance",\n            "i": "http://www.w3.org/2001/XMLSchema-instance"\n        }\n\n        xml.attrib["xmlns:i"] = "http://www.w3.org/2001/XMLSchema-instance"\n        xml.attrib["xmlns"] = "http://schemas.microsoft.com/windowsazure"\n        '
    xml = ET.Element(document_element_name)
    xml.set('xmlns', 'http://schemas.microsoft.com/windowsazure')
    if inner_xml is not None:
        xml.append(inner_xml)
    return xml