from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.linode import get_user_agent
def initialise_module():
    """Initialise the module parameter specification."""
    return AnsibleModule(argument_spec=dict(label=dict(type='str', required=True), state=dict(type='str', required=True, choices=['present', 'absent']), access_token=dict(type='str', required=True, no_log=True, fallback=(env_fallback, ['LINODE_ACCESS_TOKEN'])), authorized_keys=dict(type='list', elements='str', no_log=False), group=dict(type='str'), image=dict(type='str'), private_ip=dict(type='bool', default=False), region=dict(type='str'), root_pass=dict(type='str', no_log=True), tags=dict(type='list', elements='str'), type=dict(type='str'), stackscript_id=dict(type='int'), stackscript_data=dict(type='dict')), supports_check_mode=False, required_one_of=(['state', 'label'],), required_together=(['region', 'image', 'type'],))