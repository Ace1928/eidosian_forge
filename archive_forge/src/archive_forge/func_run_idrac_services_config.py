from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import json
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def run_idrac_services_config(idrac, module):
    """
    Get Lifecycle Controller status

    Keyword arguments:
    idrac  -- iDRAC handle
    module -- Ansible module
    """
    idrac.use_redfish = True
    share_path = tempfile.gettempdir() + os.sep
    upd_share = file_share_manager.create_share_obj(share_path=share_path, isFolder=True)
    if not upd_share.IsValid:
        module.fail_json(msg='Unable to access the share. Ensure that the share name, share mount, and share credentials provided are correct.')
    set_liason = idrac.config_mgr.set_liason_share(upd_share)
    if set_liason['Status'] == 'Failed':
        try:
            message = set_liason['Data']['Message']
        except (IndexError, KeyError):
            message = set_liason['Message']
        module.fail_json(msg=message)
    if module.params['enable_web_server'] is not None:
        idrac.config_mgr.configure_web_server(enable_web_server=Enable_WebServerTypes[module.params['enable_web_server']])
    if module.params['http_port'] is not None:
        idrac.config_mgr.configure_web_server(http_port=module.params['http_port'])
    if module.params['https_port'] is not None:
        idrac.config_mgr.configure_web_server(https_port=module.params['https_port'])
    if module.params['timeout'] is not None:
        idrac.config_mgr.configure_web_server(timeout=module.params['timeout'])
    if module.params['ssl_encryption'] is not None:
        idrac.config_mgr.configure_web_server(ssl_encryption=SSLEncryptionBitLength_WebServerTypes[module.params['ssl_encryption']])
    if module.params['tls_protocol'] is not None:
        idrac.config_mgr.configure_web_server(tls_protocol=TLSProtocol_WebServerTypes[module.params['tls_protocol']])
    if module.params['snmp_enable'] is not None:
        idrac.config_mgr.configure_snmp(snmp_enable=AgentEnable_SNMPTypes[module.params['snmp_enable']])
    if module.params['community_name'] is not None:
        idrac.config_mgr.configure_snmp(community_name=module.params['community_name'])
    if module.params['snmp_protocol'] is not None:
        idrac.config_mgr.configure_snmp(snmp_protocol=SNMPProtocol_SNMPTypes[module.params['snmp_protocol']])
    if module.params['alert_port'] is not None:
        idrac.config_mgr.configure_snmp(alert_port=module.params['alert_port'])
    if module.params['discovery_port'] is not None:
        idrac.config_mgr.configure_snmp(discovery_port=module.params['discovery_port'])
    if module.params['trap_format'] is not None:
        idrac.config_mgr.configure_snmp(trap_format=module.params['trap_format'])
    if module.params['ipmi_lan'] is not None:
        ipmi_option = module.params.get('ipmi_lan')
        community_name = ipmi_option.get('community_name')
        if community_name is not None:
            idrac.config_mgr.configure_snmp(ipmi_community=community_name)
    if module.check_mode:
        status = idrac.config_mgr.is_change_applicable()
        if status.get('changes_applicable'):
            module.exit_json(msg='Changes found to commit!', changed=True)
        else:
            module.exit_json(msg='No changes found to commit!')
    else:
        return idrac.config_mgr.apply_changes(reboot=False)