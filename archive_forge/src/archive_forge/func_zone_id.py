from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def zone_id(self, name):
    """ Search for zone id by zone name.

        Returns:
            the zone id, or send a module Fail signal if zone not found.
        """
    zone = self.manageiq.find_collection_resource_by('zones', name=name)
    if not zone:
        self.module.fail_json(msg='zone %s does not exist in manageiq' % name)
    return zone['id']