from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def retrieve_group_instances(client, module, group_id):
    wait_timeout = module.params.get('wait_timeout')
    wait_for_instances = module.params.get('wait_for_instances')
    health_check_type = module.params.get('health_check_type')
    if wait_timeout is None:
        wait_timeout = 300
    wait_timeout = time.time() + wait_timeout
    target = module.params.get('target')
    state = module.params.get('state')
    instances = list()
    if state == 'present' and group_id is not None and (wait_for_instances is True):
        is_amount_fulfilled = False
        while is_amount_fulfilled is False and wait_timeout > time.time():
            instances = list()
            amount_of_fulfilled_instances = 0
            if health_check_type is not None:
                healthy_instances = client.get_instance_healthiness(group_id=group_id)
                for healthy_instance in healthy_instances:
                    if healthy_instance.get('healthStatus') == 'HEALTHY':
                        amount_of_fulfilled_instances += 1
                        instances.append(healthy_instance)
            else:
                active_instances = client.get_elastigroup_active_instances(group_id=group_id)
                for active_instance in active_instances:
                    if active_instance.get('private_ip') is not None:
                        amount_of_fulfilled_instances += 1
                        instances.append(active_instance)
            if amount_of_fulfilled_instances >= target:
                is_amount_fulfilled = True
            time.sleep(10)
    return instances