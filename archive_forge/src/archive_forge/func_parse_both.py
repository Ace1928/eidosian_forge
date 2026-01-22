from __future__ import absolute_import, division, print_function
import re
import time
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_both(configobj, name, address_family='global'):
    rd_pattern = re.compile('(?P<rd>.+:.+)')
    matches = list()
    export_match = None
    import_match = None
    if address_family == 'global':
        export_match = parse_export(configobj, name)
        import_match = parse_import(configobj, name)
    elif address_family == 'ipv4':
        export_match = parse_export_ipv4(configobj, name)
        import_match = parse_import_ipv4(configobj, name)
    elif address_family == 'ipv6':
        export_match = parse_export_ipv6(configobj, name)
        import_match = parse_import_ipv6(configobj, name)
    if import_match and export_match:
        for ex in export_match:
            exrd = rd_pattern.search(ex)
            exrd = exrd.groupdict().get('rd')
            for im in import_match:
                imrd = rd_pattern.search(im)
                imrd = imrd.groupdict().get('rd')
                if exrd == imrd:
                    matches.extend([exrd]) if exrd not in matches else None
                    matches.extend([imrd]) if imrd not in matches else None
    return matches