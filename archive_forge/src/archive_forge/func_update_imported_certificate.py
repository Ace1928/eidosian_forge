import base64
import re  # regex library
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.acm import ACMServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_imported_certificate(client, module, acm, old_cert, desired_tags):
    """
    Update the existing certificate that was previously imported in ACM.
    """
    module.debug('Existing certificate found in ACM')
    if 'tags' not in old_cert or 'Name' not in old_cert['tags']:
        module.fail_json(msg='Internal error, unsure which certificate to update', certificate=old_cert)
    if module.params.get('name_tag') is not None and old_cert['tags']['Name'] != module.params.get('name_tag'):
        module.fail_json(msg='Internal error, Name tag does not match', certificate=old_cert)
    if 'certificate' not in old_cert:
        module.fail_json(msg='Internal error, unsure what the existing cert in ACM is', certificate=old_cert)
    cert_arn = None
    same = True
    if module.params.get('certificate') is not None:
        same &= chain_compare(module, old_cert['certificate'], module.params['certificate'])
        if module.params['certificate_chain']:
            same &= chain_compare(module, old_cert['certificate_chain'], module.params['certificate_chain'])
        else:
            same &= chain_compare(module, old_cert['certificate_chain'], module.params['certificate'])
    if same:
        module.debug('Existing certificate in ACM is the same')
        cert_arn = old_cert['certificate_arn']
        changed = False
    else:
        absent_args = ['certificate', 'name_tag', 'private_key']
        if sum([module.params[a] is not None for a in absent_args]) < 3:
            module.fail_json(msg="When importing a certificate, all of 'name_tag', 'certificate' and 'private_key' must be specified")
        module.debug('Existing certificate in ACM is different, overwriting')
        changed = True
        if module.check_mode:
            cert_arn = old_cert['certificate_arn']
        else:
            cert_arn = acm.import_certificate(client, module, certificate=module.params['certificate'], private_key=module.params['private_key'], certificate_chain=module.params['certificate_chain'], arn=old_cert['certificate_arn'], tags=desired_tags)
    return (changed, cert_arn)