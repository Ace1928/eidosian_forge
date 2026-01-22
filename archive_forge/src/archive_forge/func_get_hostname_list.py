from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_hostname_list(module):
    hostnames = module.params.get('hostnames')
    count = module.params.get('count')
    count_offset = module.params.get('count_offset')
    if isinstance(hostnames, str):
        hostnames = listify_string_name_or_id(hostnames)
    if not isinstance(hostnames, list):
        raise Exception('name %s is not convertible to list' % hostnames)
    hostnames = [h.strip() for h in hostnames]
    if len(hostnames) > 1 and count > 1:
        _msg = 'If you set count>1, you should only specify one hostname with the %d formatter, not a list of hostnames.'
        raise Exception(_msg)
    if len(hostnames) == 1 and count > 0:
        hostname_spec = hostnames[0]
        count_range = range(count_offset, count_offset + count)
        if re.search('%\\d{0,2}d', hostname_spec):
            hostnames = [hostname_spec % i for i in count_range]
        elif count > 1:
            hostname_spec = '%s%%02d' % hostname_spec
            hostnames = [hostname_spec % i for i in count_range]
    for hn in hostnames:
        if not is_valid_hostname(hn):
            raise Exception("Hostname '%s' does not seem to be valid" % hn)
    if len(hostnames) > MAX_DEVICES:
        raise Exception('You specified too many hostnames, max is %d' % MAX_DEVICES)
    return hostnames