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
def linux_configuration_to_xml(configuration, xml):
    AzureXmlSerializer.data_to_xml([('ConfigurationSetType', configuration.configuration_set_type)], xml)
    AzureXmlSerializer.data_to_xml([('HostName', configuration.host_name)], xml)
    AzureXmlSerializer.data_to_xml([('UserName', configuration.user_name)], xml)
    AzureXmlSerializer.data_to_xml([('UserPassword', configuration.user_password)], xml)
    AzureXmlSerializer.data_to_xml([('DisableSshPasswordAuthentication', configuration.disable_ssh_password_authentication, _lower)], xml)
    if configuration.ssh is not None:
        ssh = ET.Element('SSH')
        pkeys = ET.Element('PublicKeys')
        kpairs = ET.Element('KeyPairs')
        ssh.append(pkeys)
        ssh.append(kpairs)
        xml.append(ssh)
        for key in configuration.ssh.public_keys:
            pkey = ET.Element('PublicKey')
            pkeys.append(pkey)
            AzureXmlSerializer.data_to_xml([('Fingerprint', key.fingerprint)], pkey)
            AzureXmlSerializer.data_to_xml([('Path', key.path)], pkey)
        for key in configuration.ssh.key_pairs:
            kpair = ET.Element('KeyPair')
            kpairs.append(kpair)
            AzureXmlSerializer.data_to_xml([('Fingerprint', key.fingerprint)], kpair)
            AzureXmlSerializer.data_to_xml([('Path', key.path)], kpair)
    if configuration.custom_data is not None:
        AzureXmlSerializer.data_to_xml([('CustomData', configuration.custom_data)], xml)
    return xml