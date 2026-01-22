from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_encoding_id(cursor, encoding):
    query = 'SELECT pg_char_to_encoding(%(encoding)s) AS encoding_id;'
    cursor.execute(query, {'encoding': encoding})
    return cursor.fetchone()['encoding_id']