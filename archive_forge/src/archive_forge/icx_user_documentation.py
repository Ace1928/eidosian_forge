from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import base64
import hashlib
from functools import partial
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_config, load_config
entry point for module execution
    