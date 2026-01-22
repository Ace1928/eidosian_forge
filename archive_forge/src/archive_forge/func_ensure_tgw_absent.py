from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_tgw_absent(self, tgw_id=None, description=None):
    """
        Will delete the tgw if a single tgw is found not yet in deleted status

        :param tgw_id:  The AWS id of the transit gateway
        :param description:  The description of the transit gateway.
        :return doct: transit gateway object
        """
    self._results['transit_gateway_id'] = None
    tgw = self.get_matching_tgw(tgw_id, description)
    if tgw is not None:
        if self._check_mode:
            self._results['changed'] = True
            return self._results
        try:
            tgw = self.delete_tgw(tgw_id=tgw['transit_gateway_id'])
            self._results['changed'] = True
            self._results['transit_gateway'] = self.get_matching_tgw(tgw_id=tgw['transit_gateway_id'], skip_deleted=False)
        except (BotoCoreError, ClientError) as e:
            self._module.fail_json_aws(e, msg='Unable to delete Transit Gateway')
    return self._results