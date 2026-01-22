from __future__ import (absolute_import, division, print_function)
from ansible.module_utils import basic
def cpg_argument_spec():
    spec = {'state': {'required': True, 'choices': ['present', 'absent'], 'type': 'str'}, 'cpg_name': {'required': True, 'type': 'str'}, 'domain': {'type': 'str'}, 'growth_increment': {'type': 'str'}, 'growth_limit': {'type': 'str'}, 'growth_warning': {'type': 'str'}, 'raid_type': {'required': False, 'type': 'str', 'choices': ['R0', 'R1', 'R5', 'R6']}, 'set_size': {'required': False, 'type': 'int'}, 'high_availability': {'type': 'str', 'choices': ['PORT', 'CAGE', 'MAG']}, 'disk_type': {'type': 'str', 'choices': ['FC', 'NL', 'SSD']}}
    spec.update(storage_system_spec)
    return spec