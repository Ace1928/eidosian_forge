from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
Check if the role exists.

    Args:
        client (cursor): Mongodb cursor on admin database.
        user (str): Role to check.
        db_name (str): Role's database.

    Returns:
        dict: when role exists, False otherwise.
    