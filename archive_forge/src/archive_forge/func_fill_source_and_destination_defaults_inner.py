from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
from ansible.module_utils._text import to_native
def fill_source_and_destination_defaults_inner(self, obj):
    addresses = obj.get('addresses') or []
    droplet_ids = obj.get('droplet_ids') or []
    droplet_ids = [str(droplet_id) for droplet_id in droplet_ids]
    load_balancer_uids = obj.get('load_balancer_uids') or []
    load_balancer_uids = [str(uid) for uid in load_balancer_uids]
    tags = obj.get('tags') or []
    data = {'addresses': addresses, 'droplet_ids': droplet_ids, 'load_balancer_uids': load_balancer_uids, 'tags': tags}
    return data