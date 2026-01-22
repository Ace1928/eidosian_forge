from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def validate_netconf_port(value, module):
    if not 1 <= value <= 65535:
        module.fail_json(msg='netconf_port must be between 1 and 65535')