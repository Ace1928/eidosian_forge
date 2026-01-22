import json
import traceback
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def set_repository_policy(self, registry_id, name, policy_text, force):
    if not self.check_mode:
        policy = self.ecr.set_repository_policy(repositoryName=name, policyText=policy_text, force=force, **build_kwargs(registry_id))
        self.changed = True
        return policy
    else:
        self.skipped = True
        if self.get_repository(registry_id, name) is None:
            printable = name
            if registry_id:
                printable = f'{registry_id}:{name}'
            raise Exception(f'could not find repository {printable}')
        return