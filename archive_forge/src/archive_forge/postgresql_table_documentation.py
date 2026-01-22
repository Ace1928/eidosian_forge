from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (

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
        