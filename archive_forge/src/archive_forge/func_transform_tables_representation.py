from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def transform_tables_representation(tbl_list):
    """Add 'public.' to names of tables where a schema identifier is absent
    and add quotes to each element.

    Args:
        tbl_list (list): List of table names.

    Returns:
        tbl_list (list): Changed list.
    """
    for i, table in enumerate(tbl_list):
        if '.' not in table:
            tbl_list[i] = pg_quote_identifier('public.%s' % table.strip(), 'table')
        else:
            tbl_list[i] = pg_quote_identifier(table.strip(), 'table')
    return tbl_list