from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
import datetime
def create_password(self, old_passwords):

    def gen_guid():
        return uuid.uuid4()
    if self.value is None:
        self.fail("when creating a new password, module parameter value can't be None")
    start_date = datetime.datetime.now(datetime.timezone.utc)
    end_date = self.end_date or start_date + relativedelta(years=1)
    value = self.value
    key_id = self.key_id or str(gen_guid())
    new_password = PasswordCredential(start_date=start_date, end_date=end_date, key_id=key_id, value=value, custom_key_identifier=None)
    old_passwords.append(new_password)
    try:
        client = self.get_graphrbac_client(self.tenant)
        app_patch_parameters = ApplicationUpdateParameters(password_credentials=old_passwords)
        client.applications.patch(self.app_object_id, app_patch_parameters)
        new_passwords = self.get_all_passwords()
        for pd in new_passwords:
            if pd.key_id == key_id:
                self.results['changed'] = True
                self.results.update(self.to_dict(pd))
    except GraphErrorException as ge:
        self.fail('failed to create new password: {0}'.format(str(ge)))