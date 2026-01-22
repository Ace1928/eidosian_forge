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
def role_to_xml(availability_set_name, data_virtual_hard_disks, network_configuration_set, os_virtual_hard_disk, vm_image_name, role_name, role_size, role_type, system_configuration_set, xml):
    AzureXmlSerializer.data_to_xml([('RoleName', role_name)], xml)
    AzureXmlSerializer.data_to_xml([('RoleType', role_type)], xml)
    config_sets = ET.Element('ConfigurationSets')
    xml.append(config_sets)
    if system_configuration_set is not None:
        config_set = ET.Element('ConfigurationSet')
        config_sets.append(config_set)
        if isinstance(system_configuration_set, WindowsConfigurationSet):
            AzureXmlSerializer.windows_configuration_to_xml(system_configuration_set, config_set)
        elif isinstance(system_configuration_set, LinuxConfigurationSet):
            AzureXmlSerializer.linux_configuration_to_xml(system_configuration_set, config_set)
    if network_configuration_set is not None:
        config_set = ET.Element('ConfigurationSet')
        config_sets.append(config_set)
        AzureXmlSerializer.network_configuration_to_xml(network_configuration_set, config_set)
    if availability_set_name is not None:
        AzureXmlSerializer.data_to_xml([('AvailabilitySetName', availability_set_name)], xml)
    if data_virtual_hard_disks is not None:
        vhds = ET.Element('DataVirtualHardDisks')
        xml.append(vhds)
        for hd in data_virtual_hard_disks:
            vhd = ET.Element('DataVirtualHardDisk')
            vhds.append(vhd)
            AzureXmlSerializer.data_to_xml([('HostCaching', hd.host_caching)], vhd)
            AzureXmlSerializer.data_to_xml([('DiskLabel', hd.disk_label)], vhd)
            AzureXmlSerializer.data_to_xml([('DiskName', hd.disk_name)], vhd)
            AzureXmlSerializer.data_to_xml([('Lun', hd.lun)], vhd)
            AzureXmlSerializer.data_to_xml([('LogicalDiskSizeInGB', hd.logical_disk_size_in_gb)], vhd)
            AzureXmlSerializer.data_to_xml([('MediaLink', hd.media_link)], vhd)
    if os_virtual_hard_disk is not None:
        hd = ET.Element('OSVirtualHardDisk')
        xml.append(hd)
        AzureXmlSerializer.data_to_xml([('HostCaching', os_virtual_hard_disk.host_caching)], hd)
        AzureXmlSerializer.data_to_xml([('DiskLabel', os_virtual_hard_disk.disk_label)], hd)
        AzureXmlSerializer.data_to_xml([('DiskName', os_virtual_hard_disk.disk_name)], hd)
        AzureXmlSerializer.data_to_xml([('MediaLink', os_virtual_hard_disk.media_link)], hd)
        AzureXmlSerializer.data_to_xml([('SourceImageName', os_virtual_hard_disk.source_image_name)], hd)
    if vm_image_name is not None:
        AzureXmlSerializer.data_to_xml([('VMImageName', vm_image_name)], xml)
    if role_size is not None:
        AzureXmlSerializer.data_to_xml([('RoleSize', role_size)], xml)
    return xml