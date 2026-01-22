from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, _load_params
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper
def get_existing_pipeline_variable(module, bitbucket):
    """
    Search for a pipeline variable

    :param module: instance of the :class:`AnsibleModule`
    :param bitbucket: instance of the :class:`BitbucketHelper`
    :return: existing variable or None if not found
    :rtype: dict or None

    Return example::

        {
            'name': 'AWS_ACCESS_OBKEY_ID',
            'value': 'x7HU80-a2',
            'type': 'pipeline_variable',
            'secured': False,
            'uuid': '{9ddb0507-439a-495a-99f3-5464f15128127}'
        }

    The `value` key in dict is absent in case of secured variable.
    """
    variables_base_url = BITBUCKET_API_ENDPOINTS['pipeline-variable-list'].format(workspace=module.params['workspace'], repo_slug=module.params['repository'])
    page = 1
    while True:
        next_url = '%s?page=%s' % (variables_base_url, page)
        info, content = bitbucket.request(api_url=next_url, method='GET')
        if info['status'] == 404:
            module.fail_json(msg='Invalid `repository` or `workspace`.')
        if info['status'] != 200:
            module.fail_json(msg='Failed to retrieve the list of pipeline variables: {0}'.format(info))
        if 'pagelen' in content and content['pagelen'] == 0:
            return None
        page += 1
        var = next(filter(lambda v: v['key'] == module.params['name'], content['values']), None)
        if var is not None:
            var['name'] = var.pop('key')
            return var