from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, replace_resource_dict
from ansible.module_utils._text import to_native
import json
import os
import base64
def key_name_from_file(filename, module):
    with open(filename, 'r') as f:
        try:
            json_data = json.loads(f.read())
            return 'projects/{project_id}/serviceAccounts/{client_email}/keys/{private_key_id}'.format(**json_data)
        except ValueError as inst:
            module.fail_json(msg='File is not a valid GCP JSON service account key')