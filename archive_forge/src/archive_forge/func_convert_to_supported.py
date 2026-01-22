from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def convert_to_supported(val):
    """Convert unsupported type to appropriate.
    Args:
        val (any) -- Any value fetched from database.
    Returns value of appropriate type.
    """
    if isinstance(val, Decimal):
        return float(val)
    elif isinstance(val, timedelta):
        return str(val)
    return val