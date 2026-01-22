from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.remote_management.lxca.common import LXCA_COMMON_ARGS, has_pylxca, connection_object

    This function invoke commands
    :param module: Ansible module object
    