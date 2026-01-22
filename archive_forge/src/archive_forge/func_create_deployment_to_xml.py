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
def create_deployment_to_xml(name, package_url, label, configuration, start_deployment, treat_warnings_as_error, extended_properties):
    return AzureXmlSerializer.doc_from_data('CreateDeployment', [('Name', name), ('PackageUrl', package_url), ('Label', label, AzureNodeDriver._encode_base64), ('Configuration', configuration), ('StartDeployment', start_deployment, _lower), ('TreatWarningsAsError', treat_warnings_as_error, _lower)], extended_properties)