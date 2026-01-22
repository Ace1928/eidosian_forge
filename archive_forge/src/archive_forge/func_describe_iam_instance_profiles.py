from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_instance_profiles
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def describe_iam_instance_profiles(module, client):
    name = module.params['name']
    prefix = module.params['path_prefix']
    profiles = []
    profiles = list_iam_instance_profiles(client, name=name, prefix=prefix)
    return [normalize_iam_instance_profile(p) for p in profiles]