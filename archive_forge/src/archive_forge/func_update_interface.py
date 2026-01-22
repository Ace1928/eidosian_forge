from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_interface(module, array, interface):
    """Modify Interface settings"""
    changed = False
    current_state = {'enabled': interface['enabled'], 'mtu': interface['mtu'], 'gateway': interface['gateway'], 'address': interface['address'], 'netmask': interface['netmask'], 'services': sorted(interface['services']), 'slaves': sorted(interface['slaves'])}
    array6 = get_array(module)
    subinterfaces = sorted(current_state['slaves'])
    if module.params['subinterfaces']:
        new_subinterfaces, dummy = _create_subinterfaces(module, array6)
        if new_subinterfaces != subinterfaces:
            subinterfaces = new_subinterfaces
        else:
            subinterfaces = current_state['slaves']
    if module.params['subordinates']:
        new_subordinates, dummy = _create_subordinates(module, array6)
        if new_subordinates != subinterfaces:
            subinterfaces = new_subordinates
        else:
            subinterfaces = current_state['slaves']
    if module.params['enabled'] != current_state['enabled']:
        enabled = module.params['enabled']
    else:
        enabled = current_state['enabled']
    if not current_state['gateway']:
        try:
            if valid_ipv4(interface['address']):
                current_state['gateway'] = None
            elif valid_ipv6(interface['address']):
                current_state['gateway'] = None
        except AttributeError:
            current_state['gateway'] = None
    if not module.params['servicelist']:
        services = sorted(interface['services'])
    else:
        services = sorted(module.params['servicelist'])
    if not module.params['address']:
        address = interface['address']
        netmask = interface['netmask']
    else:
        if module.params['gateway'] and module.params['gateway'] not in ['0.0.0.0', '::']:
            if module.params['gateway'] not in IPNetwork(module.params['address']):
                module.fail_json(msg='Gateway and subnet are not compatible.')
        if not module.params['gateway'] and interface['gateway'] not in [None, IPNetwork(module.params['address'])]:
            module.fail_json(msg='Gateway and subnet are not compatible.')
        address = str(module.params['address'].split('/', 1)[0])
        if address in ['0.0.0.0', '::']:
            address = None
    if not module.params['mtu']:
        mtu = interface['mtu']
    elif not 1280 <= module.params['mtu'] <= 9216:
        module.fail_json(msg='MTU {0} is out of range (1280 to 9216)'.format(module.params['mtu']))
    else:
        mtu = module.params['mtu']
    if module.params['address']:
        if valid_ipv4(address):
            netmask = str(IPNetwork(module.params['address']).netmask)
        else:
            netmask = str(module.params['address'].split('/', 1)[1])
        if netmask in ['0.0.0.0', '0']:
            netmask = None
    else:
        netmask = interface['netmask']
    if not module.params['gateway']:
        gateway = interface['gateway']
    elif module.params['gateway'] in ['0.0.0.0', '::']:
        gateway = None
    elif valid_ipv4(address):
        cidr = str(IPAddress(netmask).netmask_bits())
        full_addr = address + '/' + cidr
        if module.params['gateway'] not in IPNetwork(full_addr):
            module.fail_json(msg='Gateway and subnet are not compatible.')
        gateway = module.params['gateway']
    else:
        gateway = module.params['gateway']
    new_state = {'enabled': enabled, 'address': address, 'mtu': mtu, 'gateway': gateway, 'netmask': netmask, 'services': sorted(services), 'slaves': sorted(subinterfaces)}
    if new_state['address']:
        if current_state['address'] and IPAddress(new_state['address']).version != IPAddress(current_state['address']).version:
            if new_state['gateway']:
                if IPAddress(new_state['gateway']).version != IPAddress(new_state['address']).version:
                    module.fail_json(msg='Changing IP protocol requires gateway to change as well.')
    if new_state != current_state:
        changed = True
        if module.params['servicelist'] and sorted(module.params['servicelist']) != interface['services']:
            api_version = array._list_available_rest_versions()
            if FC_ENABLE_API in api_version:
                if HAS_PYPURECLIENT:
                    if not module.check_mode:
                        network = NetworkInterfacePatch(services=module.params['servicelist'])
                        res = array6.patch_network_interfaces(names=[module.params['name']], network=network)
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to update interface service list {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
                else:
                    module.warn_json('Servicelist not updated as pypureclient module is required')
        if ('management' in interface['services'] or 'app' in interface['services']) and address in ['0.0.0.0/0', '::/0']:
            module.fail_json(msg='Removing IP address from a management or app port is not supported')
        if not module.check_mode:
            try:
                array.set_network_interface(interface['name'], enabled=new_state['enabled'])
                if new_state['gateway'] is not None:
                    array.set_network_interface(interface['name'], address=new_state['address'], mtu=new_state['mtu'], netmask=new_state['netmask'], gateway=new_state['gateway'])
                    if current_state['slaves'] != new_state['slaves'] and new_state['slaves'] != []:
                        array.set_network_interface(interface['name'], subinterfacelist=new_state['slaves'])
                else:
                    if valid_ipv4(new_state['address']):
                        empty_gateway = '0.0.0.0'
                    else:
                        empty_gateway = '::'
                    array.set_network_interface(interface['name'], address=new_state['address'], mtu=new_state['mtu'], netmask=new_state['netmask'], gateway=empty_gateway)
                    if current_state['slaves'] != new_state['slaves'] and new_state['slaves'] != []:
                        array.set_network_interface(interface['name'], subinterfacelist=new_state['slaves'])
            except Exception:
                module.fail_json(msg='Failed to change settings for interface {0}.'.format(interface['name']))
    module.exit_json(changed=changed)