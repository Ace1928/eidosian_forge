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
def get_template_ids(self, template_list):
    template_ids = []
    if template_list is None or len(template_list) == 0:
        return template_ids
    for template in template_list:
        template_list = self._zapi.template.get({'output': 'extend', 'filter': {'host': template}})
        if len(template_list) < 1:
            continue
        else:
            template_id = template_list[0]['templateid']
            template_ids.append({'templateid': template_id})
    return template_ids