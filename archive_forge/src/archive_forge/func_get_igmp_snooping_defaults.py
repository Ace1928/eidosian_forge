from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_igmp_snooping_defaults():
    group_timeout = 'dummy'
    report_supp = True
    link_local_grp_supp = True
    v3_report_supp = False
    snooping = True
    args = dict(snooping=snooping, link_local_grp_supp=link_local_grp_supp, report_supp=report_supp, v3_report_supp=v3_report_supp, group_timeout=group_timeout)
    default = dict(((param, value) for param, value in args.items() if value is not None))
    return default