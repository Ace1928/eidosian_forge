from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def is_valid_bdf_multiplier(multiplier):
    """check if the detect_multiplier is valid"""
    if multiplier < 3 or multiplier > 50:
        return False
    return True