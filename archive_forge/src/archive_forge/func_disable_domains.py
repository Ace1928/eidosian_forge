from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def disable_domains(self):
    """Delete all existing domains on storage system."""
    for domain_id in self.existing_domain_ids:
        self.delete_domain(domain_id)