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
def data_virtual_hard_disk_to_xml(host_caching, disk_label, disk_name, lun, logical_disk_size_in_gb, media_link, source_media_link):
    return AzureXmlSerializer.doc_from_data('DataVirtualHardDisk', [('HostCaching', host_caching), ('DiskLabel', disk_label), ('DiskName', disk_name), ('Lun', lun), ('LogicalDiskSizeInGB', logical_disk_size_in_gb), ('MediaLink', media_link), ('SourceMediaLink', source_media_link)])