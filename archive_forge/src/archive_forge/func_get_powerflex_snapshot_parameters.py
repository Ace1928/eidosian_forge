from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def get_powerflex_snapshot_parameters():
    """This method provide parameter required for the Ansible snapshot
    module on PowerFlex"""
    return dict(snapshot_name=dict(), snapshot_id=dict(), vol_name=dict(), vol_id=dict(), read_only=dict(required=False, type='bool'), size=dict(required=False, type='int'), cap_unit=dict(choices=['GB', 'TB']), snapshot_new_name=dict(), allow_multiple_mappings=dict(required=False, type='bool'), sdc=dict(type='list', elements='dict', options=dict(sdc_id=dict(), sdc_ip=dict(), sdc_name=dict(), access_mode=dict(choices=['READ_WRITE', 'READ_ONLY', 'NO_ACCESS']), bandwidth_limit=dict(type='int'), iops_limit=dict(type='int'))), desired_retention=dict(type='int'), retention_unit=dict(choices=['hours', 'days']), remove_mode=dict(choices=['ONLY_ME', 'INCLUDING_DESCENDANTS']), sdc_state=dict(choices=['mapped', 'unmapped']), state=dict(required=True, type='str', choices=['present', 'absent']))