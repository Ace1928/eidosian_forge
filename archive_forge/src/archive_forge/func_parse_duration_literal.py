from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
from ansible.module_utils.common.text.converters import to_native
def parse_duration_literal(value, extended=False):
    duration = 0.0
    if value == 'INF':
        return duration
    lookup = (EXTENDED_DURATION_REGEX if extended else DURATION_REGEX).findall(value)
    for duration_literal in lookup:
        filtered_literal = list(filter(None, duration_literal))
        duration_val = float(filtered_literal[0])
        duration += duration_val * DURATION_UNIT_NANOSECS[filtered_literal[1]]
    return duration