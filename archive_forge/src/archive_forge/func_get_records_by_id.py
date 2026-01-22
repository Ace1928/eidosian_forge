from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def get_records_by_id(self):
    if self.records_by_id:
        return (False, [self.records_by_id])
    else:
        return (False, [])