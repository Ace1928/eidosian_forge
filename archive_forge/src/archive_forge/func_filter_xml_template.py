from __future__ import absolute_import, division, print_function
import json
import traceback
import re
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def filter_xml_template(self, template_xml):
    """Filters out keys from XML template that may wary between exports (e.g date or version) and
        keys that are not imported via this module.

        It is advised that provided XML template exactly matches XML structure used by Zabbix"""
    parsed_xml_root = self.load_xml_template(template_xml.strip())
    keep_keys = ['graphs', 'templates', 'triggers', 'value_maps']
    for node in list(parsed_xml_root):
        if node.tag not in keep_keys:
            parsed_xml_root.remove(node)
    for template in list(parsed_xml_root.find('templates')):
        for element in list(template):
            if element.text is None and len(list(element)) == 0:
                template.remove(element)
    xml_root_text = list((line.strip() for line in ET.tostring(parsed_xml_root, encoding='utf8', method='xml').decode().split('\n')))
    return ''.join(xml_root_text)