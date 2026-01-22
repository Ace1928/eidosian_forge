from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_gateway_vtl(self, gateway):
    try:
        response = self.client.list_tapes(Limit=100)
        gateway['tapes'] = []
        marker = self._read_gateway_tape_response(gateway['tapes'], response)
        while marker is not None:
            response = self.client.list_tapes(Marker=marker, Limit=100)
            marker = self._read_gateway_tape_response(gateway['tapes'], response)
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg="Couldn't list storage gateway tapes")