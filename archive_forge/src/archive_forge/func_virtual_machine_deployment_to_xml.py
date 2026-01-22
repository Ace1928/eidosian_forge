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
def virtual_machine_deployment_to_xml(deployment_name, deployment_slot, label, role_name, system_configuration_set, os_virtual_hard_disk, role_type, network_configuration_set, availability_set_name, data_virtual_hard_disks, role_size, virtual_network_name, vm_image_name):
    doc = AzureXmlSerializer.doc_from_xml('Deployment')
    AzureXmlSerializer.data_to_xml([('Name', deployment_name)], doc)
    AzureXmlSerializer.data_to_xml([('DeploymentSlot', deployment_slot)], doc)
    AzureXmlSerializer.data_to_xml([('Label', label)], doc)
    role_list = ET.Element('RoleList')
    role = ET.Element('Role')
    role_list.append(role)
    doc.append(role_list)
    AzureXmlSerializer.role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, system_configuration_set, role)
    if virtual_network_name is not None:
        doc.append(AzureXmlSerializer.data_to_xml([('VirtualNetworkName', virtual_network_name)]))
    result = ensure_string(ET.tostring(doc, encoding='utf-8'))
    return result