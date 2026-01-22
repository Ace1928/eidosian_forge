from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
def k8s_config_resource_name(resource):
    """
    Generate resource name for the given resource of type ConfigMap, Secret
    """
    try:
        return resource['metadata']['name'] + '-' + generate_hash(resource)
    except KeyError:
        raise AnsibleFilterError('resource must have a metadata.name key to generate a resource name')