from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_server_certificate(current_cert):
    changed = False
    cert = module.params.get('cert')
    cert_chain = module.params.get('cert_chain')
    if not _compare_cert(cert, current_cert.get('certificate_body', None)):
        module.fail_json(msg='Modifying the certificate body is not supported by AWS')
    if not _compare_cert(cert_chain, current_cert.get('certificate_chain', None)):
        module.fail_json(msg='Modifying the chaining certificate is not supported by AWS')
    if module.check_mode:
        return changed
    return changed