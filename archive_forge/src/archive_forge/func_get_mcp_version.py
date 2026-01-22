from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib  # noqa: F401, pylint: disable=unused-import
from ansible.module_utils.six.moves import configparser
from os.path import expanduser
from uuid import UUID
def get_mcp_version(self, location):
    """
        Get the MCP version for the specified location.
        """
    location = self.driver.ex_get_location_by_id(location)
    if MCP_2_LOCATION_NAME_PATTERN.match(location.name):
        return '2.0'
    return '1.0'