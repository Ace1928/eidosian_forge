from __future__ import absolute_import, division, print_function
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
 Ensures user is absent

    Returns (msg, changed) 