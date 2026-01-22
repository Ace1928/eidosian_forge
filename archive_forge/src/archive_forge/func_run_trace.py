from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def run_trace(module, blade):
    """Run network trace"""
    trace_fact = {}
    if module.params['source'] and module.params['component']:
        res = blade.get_network_interfaces_trace(port=module.params['port'], destination=module.params['destination'], component=module.params['component'], discover_mtu=module.params['discover_mtu'], source=module.params['source'], fragment_packet=module.params['fragment'], method=module.params['method'], resolve_hostname=module.params['resolve'])
    elif module.params['source'] and (not module.params['component']):
        res = blade.get_network_interfaces_trace(port=module.params['port'], destination=module.params['destination'], discover_mtu=module.params['discover_mtu'], source=module.params['source'], fragment_packet=module.params['fragment'], method=module.params['method'], resolve_hostname=module.params['resolve'])
    elif not module.params['source'] and module.params['component']:
        res = blade.get_network_interfaces_trace(port=module.params['port'], destination=module.params['destination'], discover_mtu=module.params['discover_mtu'], component=module.params['component'], fragment_packet=module.params['fragment'], method=module.params['method'], resolve_hostname=module.params['resolve'])
    else:
        res = blade.get_network_interfaces_trace(port=module.params['port'], destination=module.params['destination'], discover_mtu=module.params['discover_mtu'], fragment_packet=module.params['fragment'], method=module.params['method'], resolve_hostname=module.params['resolve'])
    if res.status_code != 200:
        module.fail_json(msg='Failed to run trace. Error: {0}'.format(res.errors[0].message))
    else:
        responses = list(res.items)
        for resp in range(0, len(responses)):
            comp_name = responses[resp].component_name.replace('.', '_')
            trace_fact[comp_name] = {'details': responses[resp].details}
    module.exit_json(changed=False, tracefact=trace_fact)