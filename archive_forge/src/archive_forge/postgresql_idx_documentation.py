from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
Drop PostgreSQL index.

        Return True if success, otherwise, return False.

        Args:
            schema (str) -- name of the index schema

        Kwargs:
            cascade (bool) -- automatically drop objects that depend on the index,
                default False
            concurrent (bool) -- build index in concurrent mode, default True
        