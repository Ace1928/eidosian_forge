from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native, to_bytes
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
Check if the user exists.

    Args:
        client (cursor): Mongodb cursor on admin database.
        user (str): User to check.
        db_name (str): User's database.

    Returns:
        dict: when user exists, False otherwise.
    