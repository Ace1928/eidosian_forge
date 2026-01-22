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
def ordered_json(self, obj):
    if isinstance(obj, dict):
        return sorted(((k, self.ordered_json(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return sorted((self.ordered_json(x) for x in obj))
    else:
        return obj