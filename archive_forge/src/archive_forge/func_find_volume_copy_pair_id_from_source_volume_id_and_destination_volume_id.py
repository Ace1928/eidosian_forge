from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request
def find_volume_copy_pair_id_from_source_volume_id_and_destination_volume_id(params):
    get_status = 'storage-systems/%s/volume-copy-jobs' % params['ssid']
    url = params['api_url'] + get_status
    rc, resp = request(url, method='GET', url_username=params['api_username'], url_password=params['api_password'], headers=HEADERS, validate_certs=params['validate_certs'])
    volume_copy_pair_id = None
    for potential_copy_pair in resp:
        if potential_copy_pair['sourceVolume'] == params['source_volume_id']:
            if potential_copy_pair['sourceVolume'] == params['source_volume_id']:
                volume_copy_pair_id = potential_copy_pair['id']
    return volume_copy_pair_id