from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_mfa_devices
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_mfa_devices
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def list_mfa_devices(connection, module):
    user_name = module.params.get('user_name')
    devices = list_iam_mfa_devices(connection, user_name)
    module.exit_json(changed=False, mfa_devices=normalize_iam_mfa_devices(devices))