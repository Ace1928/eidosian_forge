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
def create_hosted_service_to_xml(service_name, label, description, location, affinity_group=None, extended_properties=None):
    if affinity_group:
        return AzureXmlSerializer.doc_from_data('CreateHostedService', [('ServiceName', service_name), ('Label', label), ('Description', description), ('AffinityGroup', affinity_group)], extended_properties)
    return AzureXmlSerializer.doc_from_data('CreateHostedService', [('ServiceName', service_name), ('Label', label), ('Description', description), ('Location', location)], extended_properties)