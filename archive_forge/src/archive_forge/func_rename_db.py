from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def rename_db(module, cursor, db, target, check_mode=False):
    source_db = db_exists(cursor, db)
    target_db = db_exists(cursor, target)
    if source_db and target_db:
        module.fail_json(msg='Both the source and the target databases exist.')
    if not source_db and target_db:
        return False
    if not source_db and (not target_db):
        module.fail_json(msg='The source and the target databases do not exist.')
    if source_db and (not target_db):
        if check_mode:
            return True
        query = 'ALTER DATABASE "%s" RENAME TO "%s"' % (db, target)
        executed_commands.append(query)
        cursor.execute(query)
        return True