from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def volume_payload(module):
    params = module.params
    drives = params.get('drives')
    capacity_bytes = params.get('capacity_bytes')
    physical_disks = []
    oem = params.get('oem')
    encrypted = params.get('encrypted')
    encryption_types = params.get('encryption_types')
    volume_type = params.get('volume_type')
    raid_type = params.get('raid_type')
    apply_time = params.get('apply_time')
    if capacity_bytes:
        capacity_bytes = int(capacity_bytes)
    if drives:
        storage_base_uri = storage_collection_map['storage_base_uri']
        physical_disks = [{'@odata.id': DRIVES_URI.format(storage_base_uri=storage_base_uri, driver_id=drive_id)} for drive_id in drives]
    raid_mapper = {'Name': params.get('name'), 'BlockSizeBytes': params.get('block_size_bytes'), 'CapacityBytes': capacity_bytes, 'OptimumIOSizeBytes': params.get('optimum_io_size_bytes'), 'Drives': physical_disks}
    raid_payload = dict([(k, v) for k, v in raid_mapper.items() if v])
    if oem:
        raid_payload.update(params.get('oem'))
    if encrypted is not None:
        raid_payload.update({'Encrypted': encrypted})
    if encryption_types:
        raid_payload.update({'EncryptionTypes': [encryption_types]})
    if volume_type:
        raid_payload.update({'RAIDType': volume_type_map.get(volume_type)})
    if raid_type:
        raid_payload.update({'RAIDType': raid_type})
    if apply_time is not None:
        raid_payload.update({'@Redfish.OperationApplyTime': apply_time})
    return raid_payload