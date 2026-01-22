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
def load_xml_template(self, template_xml):
    try:
        return ET.fromstring(template_xml)
    except ET.ParseError as e:
        self._module.fail_json(msg='Invalid XML provided', details=to_native(e), exception=traceback.format_exc())