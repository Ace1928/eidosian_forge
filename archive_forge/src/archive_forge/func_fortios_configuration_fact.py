from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from urllib.parse import quote
import copy
import traceback
def fortios_configuration_fact(params, fos):
    isValid, result = validate_mkey(params)
    if not isValid:
        return (True, False, result)
    selector = params['selector']
    selector_params = params['params']
    mkey_name = MODULE_MKEY_DEFINITONS[selector]['mkey']
    mkey_value = selector_params.get(mkey_name) if selector_params else None
    [path, name] = selector.split('_')
    url_params = dict()
    if params['filters'] and len(params['filters']):
        filter_body = quote(params['filters'][0])
        for filter_item in params['filters'][1:]:
            filter_body = '%s&filter=%s' % (filter_body, quote(filter_item))
        url_params['filter'] = filter_body
    if params['sorters'] and len(params['sorters']):
        sorter_body = params['sorters'][0]
        for sorter_item in params['sorters'][1:]:
            sorter_body = '%s&sort=%s' % (sorter_body, sorter_item)
        url_params['sort'] = sorter_body
    if params['formatters'] and len(params['formatters']):
        formatter_body = params['formatters'][0]
        for formatter_item in params['formatters'][1:]:
            formatter_body = '%s|%s' % (formatter_body, formatter_item)
        url_params['format'] = formatter_body
    fact = None
    if mkey_value:
        fact = fos.get(path, name, vdom=params['vdom'], mkey=mkey_value, parameters=url_params)
    else:
        fact = fos.get(path, name, vdom=params['vdom'], parameters=url_params)
    target_playbook = []
    selector = selector.replace('.', '_').replace('-', '_')
    results = fact.get('results') if isinstance(fact.get('results'), list) else [fact.get('results')]
    for element in PLAYBOOK_BASIC_CONFIG:
        copied_element = copy.deepcopy(element)
        copied_element.update({'tasks': [{'fortios_' + selector: {'vdom': '{{ vdom }}', 'access_token': '{{ fortios_access_token }}', 'state': 'present', selector: {k: v for k, v in flatten_multilists_attributes(preprocess_to_valid_data(result), selector).items() if k not in EXCLUDED_LIST}}} for result in results]})
        target_playbook.append(copied_element)
    with open(params['output_path'] + '/' + selector + '_playbook.yml', 'w') as f:
        yaml.dump(target_playbook, f, sort_keys=False)
    return (not is_successful_status(fact), False, fact)