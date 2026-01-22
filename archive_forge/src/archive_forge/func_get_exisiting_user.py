from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def get_exisiting_user(self, client):
    ad_user = None
    try:
        if self.user_principal_name is not None:
            ad_user = client.users.get(self.user_principal_name)
        elif self.object_id is not None:
            ad_user = client.users.get(self.object_id)
        elif self.attribute_name is not None and self.attribute_value is not None:
            try:
                ad_user = list(client.users.list(filter="{0} eq '{1}'".format(self.attribute_name, self.attribute_value)))[0]
            except GraphErrorException as e:
                try:
                    ad_user = list(client.users.list(filter="{0}/any(c:c eq '{1}')".format(self.attribute_name, self.attribute_value)))[0]
                except GraphErrorException as sub_e:
                    raise
        elif self.odata_filter is not None:
            ad_user = list(client.users.list(filter=self.odata_filter))[0]
    except GraphErrorException as e:
        err_msg = str(e)
        if err_msg == "Resource '{0}' does not exist or one of its queried reference-property objects are not present.".format(self.user_principal_name):
            ad_user = None
        else:
            raise
    return ad_user