from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def build_request_body(self):
    """Build the request body."""
    self.body.update({'id': self.id, 'groupAttributes': self.group_attributes, 'ldapUrl': self.server, 'names': self.names, 'roleMapCollection': []})
    if self.search_base:
        self.body.update({'searchBase': self.search_base})
    if self.user_attribute:
        self.body.update({'userAttribute': self.user_attribute})
    if self.bind_user and self.bind_password:
        self.body.update({'bindLookupUser': {'password': self.bind_password, 'user': self.bind_user}})
    if self.role_mappings:
        for regex, names in self.role_mappings.items():
            for name in names:
                self.body['roleMapCollection'].append({'groupRegex': regex, 'ignorecase': True, 'name': name})