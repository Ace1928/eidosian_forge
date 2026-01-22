from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def check_duplicate_cert(new_cert):
    orig_cert_names = list((c['ServerCertificateName'] for c in _list_server_certficates()))
    for cert_name in orig_cert_names:
        cert = get_server_certificate(cert_name)
        if not cert:
            continue
        cert_body = cert.get('certificate_body', None)
        if not _compare_cert(new_cert, cert_body):
            continue
        module.fail_json(changed=False, msg=f'This certificate already exists under the name {cert_name} and dup_ok=False', duplicate_cert=cert)