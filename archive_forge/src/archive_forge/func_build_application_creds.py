from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def build_application_creds(self, password=None, key_value=None, key_type=None, key_usage=None, start_date=None, end_date=None, key_description=None):
    if password and key_value:
        self.fail('specify either password or key_value, but not both.')
    if not start_date:
        start_date = datetime.datetime.utcnow()
    elif isinstance(start_date, str):
        start_date = dateutil.parser.parse(start_date)
    if not end_date:
        end_date = start_date + relativedelta(years=1) - relativedelta(hours=24)
    elif isinstance(end_date, str):
        end_date = dateutil.parser.parse(end_date)
    custom_key_id = None
    if key_description and password:
        custom_key_id = self.encode_custom_key_description(key_description)
    key_type = key_type or 'AsymmetricX509Cert'
    key_usage = key_usage or 'Verify'
    password_creds = None
    key_creds = None
    if password:
        password_creds = [PasswordCredential(start_date=start_date, end_date=end_date, key_id=str(self.gen_guid()), value=password, custom_key_identifier=custom_key_id)]
    elif key_value:
        key_creds = [KeyCredential(start_date=start_date, end_date=end_date, key_id=str(self.gen_guid()), value=key_value, usage=key_usage, type=key_type, custom_key_identifier=custom_key_id)]
    return (password_creds, key_creds)