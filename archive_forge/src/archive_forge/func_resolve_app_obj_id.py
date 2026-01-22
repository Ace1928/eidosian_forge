from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
import datetime
def resolve_app_obj_id(self):
    try:
        if self.app_object_id is not None:
            return
        elif self.app_id or self.service_principal_object_id:
            if not self.app_id:
                sp = self.client.service_principals.get(self.service_principal_object_id)
                self.app_id = sp.app_id
            if not self.app_id:
                self.fail("can't resolve app via service principal object id {0}".format(self.service_principal_object_id))
            result = list(self.client.applications.list(filter="appId eq '{0}'".format(self.app_id)))
            if result:
                self.app_object_id = result[0].object_id
            else:
                self.fail("can't resolve app via app id {0}".format(self.app_id))
        else:
            self.fail('one of the [app_id, app_object_id, service_principal_id] must be set')
    except GraphErrorException as ge:
        self.fail('error in resolve app_object_id {0}'.format(str(ge)))