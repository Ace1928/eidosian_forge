import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def check_and_update_frozen_vm_configs_in_provider_section(config, head_node_config, worker_node_configs):
    provider_config = config['provider']
    vsphere_config = provider_config['vsphere_config']
    validate_frozen_vm_configs(vsphere_config['frozen_vm'])
    head_node_config['frozen_vm'] = vsphere_config['frozen_vm']
    for worker_node_config in worker_node_configs:
        worker_node_config['frozen_vm'] = {}
        worker_frozen_vm_cfg = worker_node_config['frozen_vm']
        if 'name' in head_node_config['frozen_vm']:
            worker_frozen_vm_cfg['name'] = head_node_config['frozen_vm']['name']
        if 'resource_pool' in head_node_config['frozen_vm']:
            worker_frozen_vm_cfg['resource_pool'] = head_node_config['frozen_vm']['resource_pool']