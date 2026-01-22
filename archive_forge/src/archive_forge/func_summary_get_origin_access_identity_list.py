from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def summary_get_origin_access_identity_list(self):
    try:
        origin_access_identities = []
        for origin_access_identity in self.list_origin_access_identities():
            oai_id = origin_access_identity['Id']
            oai_full_response = self.get_origin_access_identity(oai_id)
            oai_summary = {'Id': oai_id, 'ETag': oai_full_response['ETag']}
            origin_access_identities.append(oai_summary)
        return {'origin_access_identities': origin_access_identities}
    except botocore.exceptions.ClientError as e:
        self.module.fail_json_aws(e, msg='Error generating summary of origin access identities')