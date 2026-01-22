from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def is_subset_option_valid(subset_options):
    if subset_options is None:
        return (True, '', '')
    if isinstance(subset_options, dict) is False:
        raise Exception('Subset options should be provided as dictionary.')
    for key, value in subset_options.items():
        if key != 'fields' and key != 'query' and (key != 'limit') and (key != 'detail'):
            return (False, key, "Valid subset option names are:'fields', 'query', 'limit', and 'detail'")
        if key == 'limit' and type(value) is not int:
            return (False, key, "Subset options 'limit' should be provided as integer.")
        if key == 'detail' and type(value) is not bool:
            return (False, key, "Subset options 'detail' should be provided as bool.")
        if key == 'fields' and type(value) is not list:
            return (False, key, "Subset options 'fields' should be provided as list.")
        if key == 'query' and type(value) is not dict:
            return (False, key, "Subset options 'query' should be provided as dict.")
    return (True, '', '')