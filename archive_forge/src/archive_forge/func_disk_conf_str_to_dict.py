from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec,
from re import compile, match, sub
from time import sleep
def disk_conf_str_to_dict(config_string):
    """
    Transform Proxmox configuration string for disk element into dictionary which has
    volume option parsed in '{ storage }:{ volume }' format and other options parsed
    in '{ option }={ value }' format. This dictionary will be compared afterward with
    attributes that user passed to this module in playbook.

    config_string examples:
      - local-lvm:vm-100-disk-0,ssd=1,discard=on,size=25G
      - local:iso/new-vm-ignition.iso,media=cdrom,size=70k
      - none,media=cdrom
    :param config_string: Retrieved from Proxmox API configuration string
    :return: Dictionary with volume option divided into parts ('volume_name', 'storage_name', 'volume') 

        and other options as key:value.
    """
    config = config_string.split(',')
    storage_volume = config.pop(0)
    if storage_volume in ['none', 'cdrom']:
        config_current = dict(volume=storage_volume, storage_name=None, volume_name=None, size=None)
    else:
        storage_volume = storage_volume.split(':')
        storage_name = storage_volume[0]
        volume_name = storage_volume[1]
        config_current = dict(volume='%s:%s' % (storage_name, volume_name), storage_name=storage_name, volume_name=volume_name)
    config.sort()
    for option in config:
        k, v = option.split('=')
        config_current[k] = v
    return config_current