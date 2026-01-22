from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.validation import check_required_together
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_config, load_config
def prefix_length_parser(prefix, mask, module):
    if '/' in prefix and mask is not None:
        module.fail_json(msg='Ambigous, specifed both length and mask')
    if '/' in prefix:
        cidr = ip_network(to_text(prefix))
        prefix = str(cidr.network_address)
        mask = str(cidr.netmask)
    return (prefix, mask)