from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import KatelloEntityAnsibleModule, PER_PAGE
def override_to_boolnone(override):
    value = None
    if isinstance(override, bool):
        value = override
    else:
        override = override.lower()
        if override == 'enabled':
            value = True
        elif override == 'disabled':
            value = False
        elif override == 'default':
            value = None
    return value