from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
def ensure_model_attrs(self):
    req_attrs = []
    for attr in self.module.params['attributes']:
        req_attrs.append(attr['name'])
    if 'Model_Handle' not in req_attrs:
        req_attrs.append('Model_Handle')
    cur_attrs = self.find_model_by_name_type(self.module.params['name'], self.module.params['type'], req_attrs)
    Model_Handle = cur_attrs.pop('Model_Handle')
    for attr in self.module.params['attributes']:
        req_name = attr['name']
        req_val = attr['value']
        if req_val == '':
            req_val = None
        if cur_attrs[req_name] != req_val:
            if self.module.check_mode:
                self.result['changed_attrs'][req_name] = req_val
                self.result['msg'] = self.success_msg
                self.result['changed'] = True
                continue
            resp = self.update_model(Model_Handle, {req_name: req_val})
    self.module.exit_json(**self.result)