import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def purge_lifecycle_policy(self, registry_id, name):
    if not self.check_mode:
        policy = self.ecr.delete_lifecycle_policy(repositoryName=name, **build_kwargs(registry_id))
        self.changed = True
        return policy
    else:
        policy = self.get_lifecycle_policy(registry_id, name)
        if policy:
            self.skipped = True
            return policy
        return None