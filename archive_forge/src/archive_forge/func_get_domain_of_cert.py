from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import is_boto3_error_code
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
def get_domain_of_cert(self, arn, **kwargs):
    """
        returns the domain name of a certificate (encoded in the public cert)
        for a given ARN A cert with that ARN must already exist
        """
    if arn is None:
        self.module.fail_json(msg='Internal error with ACM domain fetching, no certificate ARN specified')
    error = f"Couldn't obtain certificate data for arn {arn}"
    cert_data = self.describe_certificate_with_backoff(certificate_arn=arn, module=self.module, error=error)
    return cert_data['DomainName']