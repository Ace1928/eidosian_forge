from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.linode import get_user_agent
def build_client(module):
    """Build a LinodeClient."""
    return LinodeClient(module.params['access_token'], user_agent=get_user_agent('linode_v4_module'))