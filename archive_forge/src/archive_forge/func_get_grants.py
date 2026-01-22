from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def get_grants(cursor, user, host):
    cursor.execute('SHOW GRANTS FOR %s@%s', (user, host))
    grants_line = list(filter(lambda x: 'ON *.*' in x[0], cursor.fetchall()))[0]
    pattern = '(?<=\\bGRANT\\b)(.*?)(?=(?:\\bON\\b))'
    grants = re.search(pattern, grants_line[0]).group().strip()
    return grants.split(', ')