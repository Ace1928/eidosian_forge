from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def patch_application(self, body):
    """Use REST application/applications san template to add one or more LUNs"""
    dummy, error = self.fail_if_no_uuid()
    if error is not None:
        return (dummy, error)
    api = 'application/applications'
    query = {'return_records': 'true'}
    return rest_generic.patch_async(self.rest_api, api, self.app_uuid, body, query)