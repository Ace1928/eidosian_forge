from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def sanitize_requires(tls_requires):
    sanitized_requires = {}
    if tls_requires:
        for key in tls_requires.keys():
            sanitized_requires[key.upper()] = tls_requires[key]
        if any((key in ['CIPHER', 'ISSUER', 'SUBJECT'] for key in sanitized_requires.keys())):
            sanitized_requires.pop('SSL', None)
            sanitized_requires.pop('X509', None)
            return sanitized_requires
        if 'X509' in sanitized_requires.keys():
            sanitized_requires = 'X509'
        else:
            sanitized_requires = 'SSL'
        return sanitized_requires
    return None