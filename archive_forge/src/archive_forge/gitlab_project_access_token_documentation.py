from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (

    @param project Project object
    @param name of the access token
    