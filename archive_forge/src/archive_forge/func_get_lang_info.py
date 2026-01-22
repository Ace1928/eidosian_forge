from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_lang_info(self):
    """Get information about current supported languages."""
    query = 'SELECT l.lanname, pg_catalog.pg_get_userbyid(l.lanowner) AS rolname, l.lanacl::text FROM pg_language AS l '
    res = self.__exec_sql(query)
    lang_dict = {}
    for i in res:
        lang_dict[i['lanname']] = dict(lanowner=i['rolname'], lanacl=i['lanacl'] if i['lanacl'] else '')
    return lang_dict