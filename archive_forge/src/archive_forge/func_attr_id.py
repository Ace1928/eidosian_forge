from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
def attr_id(self, name):
    """
        Get attribute hex ID
        :param name: The name of the attribute to retrieve the hex ID for
        :type name: str
        :returns: Translated hex ID of name, or None if no translation found
        :rtype: str or None
        """
    try:
        return self.attr_map[name]
    except KeyError:
        return None