from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.common.text.converters import to_native
Set service running state as needed.

    Takes into account the fact that a service may not be loaded (no supervise directory) in
    which case it is 'stopped' as far as the service manager is concerned. No status information
    can be obtained and the service can only be 'started'.
    