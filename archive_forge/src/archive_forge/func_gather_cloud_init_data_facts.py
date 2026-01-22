from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def gather_cloud_init_data_facts(module):
    res = {'cloud_init_data_facts': dict()}
    for i in ['result', 'status']:
        filter = module.params.get('filter')
        if filter is None or filter == i:
            res['cloud_init_data_facts'][i] = dict()
            json_file = os.path.join(CLOUD_INIT_PATH, i + '.json')
            if os.path.exists(json_file):
                f = open(json_file, 'rb')
                contents = to_text(f.read(), errors='surrogate_or_strict')
                f.close()
                if contents:
                    res['cloud_init_data_facts'][i] = module.from_json(contents)
    return res