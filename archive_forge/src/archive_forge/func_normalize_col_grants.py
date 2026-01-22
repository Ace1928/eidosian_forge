from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def normalize_col_grants(privileges):
    """Fix and sort grants on columns in privileges list

    Make ['SELECT (A, B)', 'INSERT (A, B)', 'DETELE']
    from ['SELECT (A', 'B)', 'INSERT (B', 'A)', 'DELETE'].
    See unit tests in tests/unit/plugins/modules/test_mysql_user.py
    """
    for grant in ('SELECT', 'UPDATE', 'INSERT', 'REFERENCES'):
        start, end = has_grant_on_col(privileges, grant)
        if start is not None:
            privileges = handle_grant_on_col(privileges, start, end)
    return privileges