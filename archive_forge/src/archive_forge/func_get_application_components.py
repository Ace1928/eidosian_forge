from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_application_components(self):
    """Use REST application/applications to get application components"""
    dummy, error = self.fail_if_no_uuid()
    if error is not None:
        return (dummy, error)
    api = 'application/applications/%s/components' % self.app_uuid
    return rest_generic.get_0_or_more_records(self.rest_api, api)