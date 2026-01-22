from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def delete_known_host(module, bitbucket, known_host_uuid):
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['known-host-detail'].format(workspace=module.params['workspace'], repo_slug=module.params['repository'], known_host_uuid=known_host_uuid), method='DELETE')
    if info['status'] == 404:
        module.fail_json(msg=error_messages['invalid_params'])
    if info['status'] != 204:
        module.fail_json(msg='Failed to delete known host `{hostname}`: {info}'.format(hostname=module.params['name'], info=info))