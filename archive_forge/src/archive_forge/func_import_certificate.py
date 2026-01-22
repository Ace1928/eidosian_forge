from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def import_certificate(self, *args, certificate, private_key, arn=None, certificate_chain=None, tags=None):
    original_arn = arn
    params = {'certificate': certificate, 'private_key': private_key, 'certificate_chain': certificate_chain, 'arn': arn, 'module': self.module, 'error': "Couldn't upload new certificate"}
    arn = self.import_certificate_with_backoff(**params)
    if original_arn and arn != original_arn:
        self.module.fail_json(msg=f'ARN changed with ACM update, from {original_arn} to {arn}')
    try:
        self.tag_certificate_with_backoff(arn, tags)
    except (BotoCoreError, ClientError) as e:
        try:
            self.delete_certificate_with_backoff(arn)
        except (BotoCoreError, ClientError):
            self.module.warn(f'Certificate {arn} exists, and is not tagged. So Ansible will not see it on the next run.')
            self.module.fail_json_aws(e, msg=f"Couldn't tag certificate {arn}, couldn't delete it either")
        self.module.fail_json_aws(e, msg=f"Couldn't tag certificate {arn}")
    return arn