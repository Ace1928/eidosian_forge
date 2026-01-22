from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def get_tls_requires(cursor, user, host):
    if user:
        if not impl.use_old_user_mgmt(cursor):
            query = "SHOW CREATE USER '%s'@'%s'" % (user, host)
        else:
            query = "SHOW GRANTS for '%s'@'%s'" % (user, host)
        cursor.execute(query)
        require_list = [tuple[0] for tuple in filter(lambda x: 'REQUIRE' in x[0], cursor.fetchall())]
        require_line = require_list[0] if require_list else ''
        pattern = '(?<=\\bREQUIRE\\b)(.*?)(?=(?:\\bPASSWORD\\b|$))'
        requires_match = re.search(pattern, require_line)
        requires = requires_match.group().strip() if requires_match else ''
        if any((requires.startswith(req) for req in ('SSL', 'X509', 'NONE'))):
            requires = requires.split()[0]
            if requires == 'NONE':
                requires = None
        else:
            import shlex
            items = iter(shlex.split(requires))
            requires = dict(zip(items, items))
        return requires or None