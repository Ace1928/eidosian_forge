from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.common.text.converters import to_native
def service_is_running(service_status):
    return service_status['DaemontoolsEncoreState'] in set(['starting', 'started', 'running'])