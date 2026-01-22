from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .retries import AWSRetry
from .waiters import get_waiter
def list_web_acls(client, module):
    try:
        if client.__class__.__name__ == 'WAF':
            return list_web_acls_with_backoff(client)
        elif client.__class__.__name__ == 'WAFRegional':
            return list_regional_web_acls_with_backoff(client)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't obtain web acls")