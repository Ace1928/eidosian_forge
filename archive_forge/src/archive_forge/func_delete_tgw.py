from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_tgw(self, tgw_id):
    """
        De;lete the transit gateway and optionally wait for status to become deleted

        :param tgw_id: The id of the transit gateway
        :return dict: transit gateway object
        """
    wait = self._module.params.get('wait')
    wait_timeout = self._module.params.get('wait_timeout')
    try:
        response = self._connection.delete_transit_gateway(TransitGatewayId=tgw_id)
    except (ClientError, BotoCoreError) as e:
        self._module.fail_json_aws(e)
    if wait:
        result = self.wait_for_status(wait_timeout=wait_timeout, tgw_id=tgw_id, status='deleted', skip_deleted=False)
    else:
        result = self.get_matching_tgw(tgw_id=tgw_id, skip_deleted=False)
    self._results['msg'] = f' Transit gateway {tgw_id} deleted'
    return result