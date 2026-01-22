from __future__ import absolute_import, division, print_function
import base64
import json
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.sops.plugins.module_utils.io import write_file
from ansible_collections.community.sops.plugins.module_utils.sops import Sops, SopsError, get_sops_argument_spec
def get_encoded_type_content(module, binary_data):
    if module.params['content_text'] is not None:
        return ('binary', module.params['content_text'].encode('utf-8'))
    if module.params['content_binary'] is not None:
        return ('binary', binary_data)
    if module.params['content_json'] is not None:
        return ('json', json.dumps(module.params['content_json']).encode('utf-8'))
    if module.params['content_yaml'] is not None:
        return ('yaml', yaml.safe_dump(module.params['content_yaml']).encode('utf-8'))
    module.fail_json(msg='Internal error: unknown content type')