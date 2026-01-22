from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def check_resource_option(resource, module):
    opts = user_input_parameters(module)
    resource = {'enterprise_project_id': resource.get('enterprise_project_id'), 'name': resource.get('name'), 'vpc_id': resource.get('vpc_id'), 'id': resource.get('id')}
    if are_different_dicts(resource, opts):
        raise Exception('Cannot change option from (%s) to (%s) for an existing security group(%s).' % (resource, opts, module.params.get('id')))