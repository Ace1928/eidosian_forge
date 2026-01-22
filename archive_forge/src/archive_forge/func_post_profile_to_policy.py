from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.intersight.plugins.module_utils.intersight import IntersightModule, intersight_argument_spec
def post_profile_to_policy(intersight, moid, resource_path, policy_name):
    options = {'http_method': 'get', 'resource_path': resource_path, 'query_params': {'$filter': "Name eq '" + policy_name + "'"}}
    response = intersight.call_api(**options)
    if response.get('Results'):
        expected_policy_moid = response['Results'][0]['Moid']
        actual_policy_moid = ''
        options = {'http_method': 'get', 'resource_path': resource_path, 'query_params': {'$filter': "Profiles/any(t: t/Moid eq '" + moid + "')"}}
        response = intersight.call_api(**options)
        if response.get('Results'):
            actual_policy_moid = response['Results'][0]['Moid']
            if actual_policy_moid != expected_policy_moid:
                if not intersight.module.check_mode:
                    options = {'http_method': 'delete', 'resource_path': resource_path + '/' + actual_policy_moid + '/Profiles', 'moid': moid}
                    intersight.call_api(**options)
                actual_policy_moid = ''
        if not actual_policy_moid:
            if not intersight.module.check_mode:
                options = {'http_method': 'post', 'resource_path': resource_path + '/' + expected_policy_moid + '/Profiles', 'body': [{'ObjectType': 'server.Profile', 'Moid': moid}]}
                intersight.call_api(**options)
            intersight.result['changed'] = True