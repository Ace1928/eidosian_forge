from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def check_name_attribute(module, attributes):
    if attributes.get('NAME'):
        import re
        if re.match('^[^#]+#*$', attributes.get('NAME')) is None:
            module.fail_json(msg="Illegal 'NAME' attribute: '" + attributes.get('NAME') + "' .Signs '#' are allowed only at the end of the name and the name cannot contain only '#'.")