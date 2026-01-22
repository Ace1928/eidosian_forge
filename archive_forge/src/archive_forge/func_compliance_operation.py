from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def compliance_operation(module, rest_obj):
    command = module.params.get('command')
    validate_names(command, module)
    validate_job_time(command, module)
    if command == 'create':
        create_baseline(module, rest_obj)
    if command == 'modify':
        modify_baseline(module, rest_obj)
    if command == 'delete':
        delete_compliance(module, rest_obj)
    if command == 'remediate':
        remediate_baseline(module, rest_obj)