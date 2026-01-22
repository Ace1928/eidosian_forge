from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_application_component_uuid(self):
    """Use REST application/applications to get component uuid
           Assume a single component per application
        """
    dummy, error = self.fail_if_no_uuid()
    if error is not None:
        return (dummy, error)
    api = 'application/applications/%s/components' % self.app_uuid
    record, error = rest_generic.get_one_record(self.rest_api, api, fields='uuid')
    if error is None and record is not None:
        return (record['uuid'], None)
    return (None, error)