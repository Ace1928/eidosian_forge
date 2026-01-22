from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def create_virtual_instance(module):
    instances = vsManager.list_instances(hostname=module.params.get('hostname'), domain=module.params.get('domain'), datacenter=module.params.get('datacenter'))
    if instances:
        return (False, None)
    if module.params.get('os_code') is not None and module.params.get('os_code') != '':
        module.params['image_id'] = ''
    elif module.params.get('image_id') is not None and module.params.get('image_id') != '':
        module.params['os_code'] = ''
        module.params['disks'] = []
    else:
        return (False, None)
    tags = module.params.get('tags')
    if isinstance(tags, list):
        tags = ','.join(map(str, module.params.get('tags')))
    instance = vsManager.create_instance(hostname=module.params.get('hostname'), domain=module.params.get('domain'), cpus=module.params.get('cpus'), memory=module.params.get('memory'), flavor=module.params.get('flavor'), hourly=module.params.get('hourly'), datacenter=module.params.get('datacenter'), os_code=module.params.get('os_code'), image_id=module.params.get('image_id'), local_disk=module.params.get('local_disk'), disks=module.params.get('disks'), ssh_keys=module.params.get('ssh_keys'), nic_speed=module.params.get('nic_speed'), private=module.params.get('private'), public_vlan=module.params.get('public_vlan'), private_vlan=module.params.get('private_vlan'), dedicated=module.params.get('dedicated'), post_uri=module.params.get('post_uri'), tags=tags)
    if instance is not None and instance['id'] > 0:
        return (True, instance)
    else:
        return (False, None)