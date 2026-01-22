from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def cloud_network(module, state, label, cidr):
    changed = False
    network = None
    networks = []
    if not pyrax.cloud_networks:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if state == 'present':
        if not cidr:
            module.fail_json(msg='missing required arguments: cidr')
        try:
            network = pyrax.cloud_networks.find_network_by_label(label)
        except pyrax.exceptions.NetworkNotFound:
            try:
                network = pyrax.cloud_networks.create(label, cidr=cidr)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
    elif state == 'absent':
        try:
            network = pyrax.cloud_networks.find_network_by_label(label)
            network.delete()
            changed = True
        except pyrax.exceptions.NetworkNotFound:
            pass
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
    if network:
        instance = dict(id=network.id, label=network.label, cidr=network.cidr)
        networks.append(instance)
    module.exit_json(changed=changed, networks=networks)