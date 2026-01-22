from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def present_datasource(module):
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    api_url = module.params['grafana_url'] + '/api/datasources'
    result = requests.post(api_url, json=module.params['dataSource'], headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    elif result.status_code == 409:
        get_id_url = requests.get(module.params['grafana_url'] + '/api/datasources/id/' + module.params['dataSource']['name'], headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        api_url = module.params['grafana_url'] + '/api/datasources/' + str(get_id_url.json()['id'])
        result = requests.put(api_url, json=module.params['dataSource'], headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        if result.status_code == 200:
            return (False, True, result.json())
        else:
            return (True, False, {'status': result.status_code, 'response': result.json()['message']})
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})