from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_tablespaces(self):
    """Get information about tablespaces."""
    opt = self.__exec_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'pg_tablespace' AND column_name = 'spcoptions'")
    if not opt:
        query = 'SELECT s.spcname, pg_catalog.pg_get_userbyid(s.spcowner) as rolname, s.spcacl::text FROM pg_tablespace AS s '
    else:
        query = 'SELECT s.spcname, pg_catalog.pg_get_userbyid(s.spcowner) as rolname, s.spcacl::text, s.spcoptions FROM pg_tablespace AS s '
    res = self.__exec_sql(query)
    ts_dict = {}
    for i in res:
        ts_name = i['spcname']
        ts_info = dict(spcowner=i['rolname'], spcacl=i['spcacl'] if i['spcacl'] else '')
        if opt:
            ts_info['spcoptions'] = i['spcoptions'] if i['spcoptions'] else []
        ts_dict[ts_name] = ts_info
    self.pg_info['tablespaces'] = ts_dict