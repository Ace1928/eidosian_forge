from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_server_certificate():
    cert = module.params.get('cert')
    key = module.params.get('key')
    cert_chain = module.params.get('cert_chain')
    if not module.params.get('dup_ok'):
        check_duplicate_cert(cert)
    path = module.params.get('path')
    name = module.params.get('name')
    params = dict(ServerCertificateName=name, CertificateBody=cert, PrivateKey=key)
    if cert_chain:
        params['CertificateChain'] = cert_chain
    if path:
        params['Path'] = path
    if module.check_mode:
        return True
    try:
        client.upload_server_certificate(aws_retry=True, **params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Failed to update server certificate {name}')
    return True