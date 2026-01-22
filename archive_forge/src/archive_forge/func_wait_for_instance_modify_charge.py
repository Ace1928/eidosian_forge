from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.alicloud_ecs import (
def wait_for_instance_modify_charge(ecs, instance_ids, charge_type, delay=10, timeout=300):
    """
    To verify instance charge type has become expected after modify instance charge type
    """
    try:
        while True:
            instances = ecs.describe_instances(instance_ids=instance_ids)
            flag = True
            for inst in instances:
                if inst and inst.instance_charge_type != charge_type:
                    flag = False
            if flag:
                return
            timeout -= delay
            time.sleep(delay)
            if timeout <= 0:
                raise Exception('Timeout Error: Waiting for instance to {0}. '.format(charge_type))
    except Exception as e:
        raise e