from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
from ansible.module_utils._text import to_native
def fill_data_defaults(self, obj):
    inbound_rules = obj.get('inbound_rules')
    if inbound_rules is None:
        inbound_rules = []
    else:
        inbound_rules = [self.fill_protocol_defaults(x) for x in inbound_rules]
        inbound_rules = [self.fill_sources_and_destinations_defaults(x, 'sources') for x in inbound_rules]
    outbound_rules = obj.get('outbound_rules')
    if outbound_rules is None:
        outbound_rules = []
    else:
        outbound_rules = [self.fill_protocol_defaults(x) for x in outbound_rules]
        outbound_rules = [self.fill_sources_and_destinations_defaults(x, 'destinations') for x in outbound_rules]
    droplet_ids = obj.get('droplet_ids') or []
    droplet_ids = [str(droplet_id) for droplet_id in droplet_ids]
    tags = obj.get('tags') or []
    data = {'name': obj.get('name'), 'inbound_rules': inbound_rules, 'outbound_rules': outbound_rules, 'droplet_ids': droplet_ids, 'tags': tags}
    return data