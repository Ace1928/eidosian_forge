import datetime
import re
from collections import OrderedDict
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def validate_viewer_certificate(self, viewer_certificate):
    try:
        if viewer_certificate is None:
            return None
        if viewer_certificate.get('cloudfront_default_certificate') and viewer_certificate.get('ssl_support_method') is not None:
            self.module.fail_json(msg='viewer_certificate.ssl_support_method should not be specified with viewer_certificate_cloudfront_default' + '_certificate set to true.')
        self.validate_attribute_with_allowed_values(viewer_certificate.get('ssl_support_method'), 'viewer_certificate.ssl_support_method', self.__valid_viewer_certificate_ssl_support_methods)
        self.validate_attribute_with_allowed_values(viewer_certificate.get('minimum_protocol_version'), 'viewer_certificate.minimum_protocol_version', self.__valid_viewer_certificate_minimum_protocol_versions)
        self.validate_attribute_with_allowed_values(viewer_certificate.get('certificate_source'), 'viewer_certificate.certificate_source', self.__valid_viewer_certificate_certificate_sources)
        viewer_certificate = change_dict_key_name(viewer_certificate, 'cloudfront_default_certificate', 'cloud_front_default_certificate')
        viewer_certificate = change_dict_key_name(viewer_certificate, 'ssl_support_method', 's_s_l_support_method')
        viewer_certificate = change_dict_key_name(viewer_certificate, 'iam_certificate_id', 'i_a_m_certificate_id')
        viewer_certificate = change_dict_key_name(viewer_certificate, 'acm_certificate_arn', 'a_c_m_certificate_arn')
        return viewer_certificate
    except Exception as e:
        self.module.fail_json_aws(e, msg='Error validating viewer certificate')