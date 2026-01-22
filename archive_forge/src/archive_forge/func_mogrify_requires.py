from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def mogrify_requires(query, params, tls_requires):
    if tls_requires:
        if isinstance(tls_requires, dict):
            k, v = zip(*tls_requires.items())
            requires_query = ' AND '.join(('%s %%s' % key for key in k))
            params += v
        else:
            requires_query = tls_requires
        query = ' REQUIRE '.join((query, requires_query))
    return (query, params)