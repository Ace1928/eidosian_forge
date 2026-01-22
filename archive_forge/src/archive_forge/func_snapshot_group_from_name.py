from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def snapshot_group_from_name(module, ssid, api_url, api_pwd, api_usr, name):
    snap_groups = 'storage-systems/%s/snapshot-groups' % ssid
    snap_groups_url = api_url + snap_groups
    ret, snapshot_groups = request(snap_groups_url, url_username=api_usr, url_password=api_pwd, headers=HEADERS, validate_certs=module.params['validate_certs'])
    snapshot_group_id = None
    for snapshot_group in snapshot_groups:
        if name == snapshot_group['label']:
            snapshot_group_id = snapshot_group['pitGroupRef']
            break
    if snapshot_group_id is None:
        module.fail_json(msg='Failed to lookup snapshot group.  Group [%s]. Id [%s].' % (name, ssid))
    return snapshot_group