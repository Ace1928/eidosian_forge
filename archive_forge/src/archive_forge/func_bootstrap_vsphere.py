import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def bootstrap_vsphere(config):
    config = copy.deepcopy(config)
    add_credentials_into_provider_section(config)
    update_vsphere_configs(config)
    check_legacy_fields(config)
    config = configure_key_pair(config)
    global_event_system.execute_callback(CreateClusterEvent.ssh_keypair_downloaded, {'ssh_key_path': config['auth']['ssh_private_key']})
    return config