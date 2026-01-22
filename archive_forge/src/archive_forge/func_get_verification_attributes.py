import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_verification_attributes(connection, module, identity, retries=0, retryDelay=10):
    for attempt in range(0, retries + 1):
        try:
            response = connection.get_identity_verification_attributes(Identities=[identity], aws_retry=True)
        except (BotoCoreError, ClientError) as e:
            module.fail_json_aws(e, msg=f'Failed to retrieve identity verification attributes for {identity}')
        identity_verification = response['VerificationAttributes']
        if identity in identity_verification:
            break
        time.sleep(retryDelay)
    if identity not in identity_verification:
        return None
    return identity_verification[identity]