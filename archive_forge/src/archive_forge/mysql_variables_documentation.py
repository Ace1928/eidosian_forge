from __future__ import absolute_import, division, print_function
import os
import warnings
from re import match
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.database import SQLParseError, mysql_quote_identifier
from ansible_collections.community.mysql.plugins.module_utils.mysql import mysql_connect, mysql_driver, mysql_driver_fail_msg, mysql_common_argument_spec
from ansible.module_utils._text import to_native
 Set a global mysql variable to a given value

    The DB driver will handle quoting of the given value based on its
    type, thus numeric strings like '3.0' or '8' are illegal, they
    should be passed as numeric literals.

    