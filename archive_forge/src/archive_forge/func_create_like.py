from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def create_like(self, src_table, including='', tblspace='', unlogged=False, params='', owner=''):
    """
        Create table like another table (with similar DDL).
        Arguments:
        src_table - source table.
        including - corresponds to optional INCLUDING expression
            in CREATE TABLE ... LIKE statement.
        params - storage params (passed by "WITH (...)" in SQL),
            comma separated.
        tblspace - tablespace.
        owner - table owner.
        unlogged - create unlogged table.
        """
    changed = False
    name = pg_quote_identifier(self.name, 'table')
    query = 'CREATE'
    if unlogged:
        query += ' UNLOGGED TABLE %s' % name
    else:
        query += ' TABLE %s' % name
    query += ' (LIKE %s' % pg_quote_identifier(src_table, 'table')
    if including:
        including = including.split(',')
        for i in including:
            query += ' INCLUDING %s' % i
    query += ')'
    if params:
        query += ' WITH (%s)' % params
    if tblspace:
        query += ' TABLESPACE "%s"' % tblspace
    if exec_sql(self, query, return_bool=True):
        changed = True
    if owner:
        changed = self.set_owner(owner)
    return changed