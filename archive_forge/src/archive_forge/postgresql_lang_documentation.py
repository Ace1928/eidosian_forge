from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
Set language owner.

    Args:
        cursor (cursor): psycopg cursor object.
        lang (str): language name.
        owner (str): name of new owner.
    