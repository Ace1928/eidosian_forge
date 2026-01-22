from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, _load_params
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def delete_pipeline_variable(module, bitbucket, variable_uuid):
    info, content = bitbucket.request(api_url=BITBUCKET_API_ENDPOINTS['pipeline-variable-detail'].format(workspace=module.params['workspace'], repo_slug=module.params['repository'], variable_uuid=variable_uuid), method='DELETE')
    if info['status'] != 204:
        module.fail_json(msg='Failed to delete pipeline variable `{name}`: {info}'.format(name=module.params['name'], info=info))