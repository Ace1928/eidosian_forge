import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_in_array(self, array_of_clusters, cluster_name, field_name='clusterArn'):
    for c in array_of_clusters:
        if c[field_name].endswith(cluster_name):
            return c
    return None