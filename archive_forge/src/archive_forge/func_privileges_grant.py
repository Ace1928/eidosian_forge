from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def privileges_grant(cursor, user, host, db_table, priv, tls_requires, maria_role=False):
    db_table = db_table.replace('%', '%%')
    priv_string = ','.join([p for p in priv if p not in ('GRANT',)])
    query = ['GRANT %s ON %s' % (priv_string, db_table)]
    if not maria_role:
        query.append('TO %s@%s')
        params = (user, host)
    else:
        query.append('TO %s')
        params = user
    if tls_requires and impl.use_old_user_mgmt(cursor):
        query, params = mogrify_requires(' '.join(query), params, tls_requires)
        query = [query]
    if 'GRANT' in priv:
        query.append('WITH GRANT OPTION')
    query = ' '.join(query)
    if isinstance(params, str):
        params = (params,)
    try:
        cursor.execute(query, params)
    except (mysql_driver.ProgrammingError, mysql_driver.OperationalError, mysql_driver.InternalError) as e:
        raise InvalidPrivsError('Error granting privileges, invalid priv string: %s , params: %s, query: %s , exception: %s.' % (priv_string, str(params), query, str(e)))