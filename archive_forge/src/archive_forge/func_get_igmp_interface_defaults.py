from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_igmp_interface_defaults():
    version = '2'
    startup_query_interval = '31'
    startup_query_count = '2'
    robustness = '2'
    querier_timeout = '255'
    query_mrt = '10'
    query_interval = '125'
    last_member_qrt = '1'
    last_member_query_count = '2'
    group_timeout = '260'
    report_llg = False
    immediate_leave = False
    args = dict(version=version, startup_query_interval=startup_query_interval, startup_query_count=startup_query_count, robustness=robustness, querier_timeout=querier_timeout, query_mrt=query_mrt, query_interval=query_interval, last_member_qrt=last_member_qrt, last_member_query_count=last_member_query_count, group_timeout=group_timeout, report_llg=report_llg, immediate_leave=immediate_leave)
    default = dict(((param, value) for param, value in args.items() if value is not None))
    return default