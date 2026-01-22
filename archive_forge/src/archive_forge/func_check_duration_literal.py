from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
from ansible.module_utils.common.text.converters import to_native
def check_duration_literal(value):
    return VALID_DURATION_REGEX.search(value) is not None