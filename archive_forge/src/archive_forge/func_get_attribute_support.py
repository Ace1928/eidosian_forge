from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def get_attribute_support(cursor):
    """Checks if the MySQL server supports user attributes.

    Args:
        cursor (cursor): DB driver cursor object.
    Returns:
        True if attributes are supported, False if they are not.
    """
    try:
        cursor.execute('SELECT attribute FROM INFORMATION_SCHEMA.USER_ATTRIBUTES LIMIT 0')
        cursor.fetchone()
    except mysql_driver.Error:
        return False
    return True