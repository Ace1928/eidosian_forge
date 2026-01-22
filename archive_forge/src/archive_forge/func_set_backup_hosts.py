from __future__ import absolute_import, division, print_function
import json
import re
import traceback
from ansible.module_utils.six import PY3
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy
def set_backup_hosts(self):
    try:
        list_of_hosts = re.sub('[[\\]]', '', self.connection.get_option('host')).split(',')
        return list_of_hosts
    except Exception:
        return []