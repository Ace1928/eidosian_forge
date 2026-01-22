from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (

    Check to see if the installed extension version has a valid update
    path to the given version.

    Return True if a valid path exists. Otherwise return False.

    Note: 'latest' is not a valid value for version here as it can be
          replaced with default_version specified in extension control file.

    Args:
      cursor (cursor) -- cursor object of psycopg library
      ext (str) -- extension name
      current_version (str) -- installed version of the extension.
      version (str) -- target extension version to update to.
    