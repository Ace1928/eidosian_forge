from __future__ import absolute_import, division, print_function
import os
import uuid
from ansible.module_utils.basic import AnsibleModule
def get_sshkey_selector(module):
    key_id = module.params.get('id')
    if key_id:
        if not is_valid_uuid(key_id):
            raise Exception('sshkey ID %s is not valid UUID' % key_id)
    selecting_fields = ['label', 'fingerprint', 'id', 'key']
    select_dict = {}
    for f in selecting_fields:
        if module.params.get(f) is not None:
            select_dict[f] = module.params.get(f)
    if module.params.get('key_file'):
        with open(module.params.get('key_file')) as _file:
            loaded_key = load_key_string(_file.read())
        select_dict['key'] = loaded_key['key']
        if module.params.get('label') is None:
            if loaded_key.get('label'):
                select_dict['label'] = loaded_key['label']

    def selector(k):
        if 'key' in select_dict:
            return k.key == select_dict['key']
        else:
            return all((select_dict[f] == getattr(k, f) for f in select_dict))
    return selector