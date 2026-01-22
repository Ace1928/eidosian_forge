from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_update_network_config(client_obj, name, state, iscsi_automatic_connection_method, iscsi_connection_rebalancing, mgmt_ip, change_name, **kwargs):
    if utils.is_null_or_empty(name):
        return (False, False, 'Create network config failed as name is not present.', {}, {})
    try:
        network_resp = client_obj.network_configs.get(id=None, name=name)
        if utils.is_null_or_empty(network_resp):
            params = utils.remove_null_args(**kwargs)
            network_resp = client_obj.network_configs.create(name=name, iscsi_automatic_connection_method=iscsi_automatic_connection_method, iscsi_connection_rebalancing=iscsi_connection_rebalancing, mgmt_ip=mgmt_ip, **params)
            return (True, True, f"Network config '{name}' created successfully.", {}, network_resp.attrs)
        else:
            if state == 'create':
                return (False, False, f"Network config '{name}' cannot be created as it is already present in given state.", {}, network_resp.attrs)
            kwargs['name'] = change_name
            changed_attrs_dict, params = utils.remove_unchanged_or_null_args(network_resp, **kwargs)
            params = utils.remove_null_args(**kwargs)
            if changed_attrs_dict.__len__() > 0:
                network_resp = client_obj.network_configs.update(id=network_resp.attrs.get('id'), name=name, iscsi_automatic_connection_method=iscsi_automatic_connection_method, iscsi_connection_rebalancing=iscsi_connection_rebalancing, mgmt_ip=mgmt_ip, **params)
                return (True, True, f"Network config '{name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, network_resp.attrs)
            else:
                return (True, False, f"Network config '{network_resp.attrs.get('name')}' already present in given state.", {}, network_resp.attrs)
    except Exception as ex:
        return (False, False, f"Network config creation failed |'{ex}'", {}, {})