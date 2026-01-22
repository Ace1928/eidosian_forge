from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def handle_global_credentials(self, response=None):
    """
        Method to convert values for create_params API when global paramters
        are passed as input.

        Parameters:
            - response: The response collected from the get_all_global_credentials_v2 API

        Returns:
            - global_credentials_all  : The dictionary containing list of IDs of various types of
                                    Global credentials.
        """
    global_credentials = self.validated_config[0].get('global_credentials')
    global_credentials_all = {}
    cli_credentials_list = global_credentials.get('cli_credentials_list')
    if cli_credentials_list:
        if not isinstance(cli_credentials_list, list):
            msg = 'Global CLI credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('cliCredential') is None:
            msg = 'Global CLI credentials are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(cli_credentials_list) > 0:
            global_credentials_all['cliCredential'] = []
            cred_len = len(cli_credentials_list)
            if cred_len > 5:
                cred_len = 5
            for cli_cred in cli_credentials_list:
                if cli_cred.get('description') and cli_cred.get('username'):
                    for cli in response.get('cliCredential'):
                        if cli.get('description') == cli_cred.get('description') and cli.get('username') == cli_cred.get('username'):
                            global_credentials_all['cliCredential'].append(cli.get('id'))
                    global_credentials_all['cliCredential'] = global_credentials_all['cliCredential'][:cred_len]
                else:
                    msg = 'Kindly ensure you include both the description and the username for the Global CLI credential to discover the devices'
                    self.discovery_specific_cred_failure(msg=msg)
    http_read_credential_list = global_credentials.get('http_read_credential_list')
    if http_read_credential_list:
        if not isinstance(http_read_credential_list, list):
            msg = 'Global HTTP read credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('httpsRead') is None:
            msg = 'Global HTTP read credentials are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(http_read_credential_list) > 0:
            global_credentials_all['httpsRead'] = []
            cred_len = len(http_read_credential_list)
            if cred_len > 5:
                cred_len = 5
            for http_cred in http_read_credential_list:
                if http_cred.get('description') and http_cred.get('username'):
                    for http in response.get('httpsRead'):
                        if http.get('description') == http.get('description') and http.get('username') == http.get('username'):
                            global_credentials_all['httpsRead'].append(http.get('id'))
                    global_credentials_all['httpsRead'] = global_credentials_all['httpsRead'][:cred_len]
                else:
                    msg = 'Kindly ensure you include both the description and the username for the Global HTTP Read credential to discover the devices'
                    self.discovery_specific_cred_failure(msg=msg)
    http_write_credential_list = global_credentials.get('http_write_credential_list')
    if http_write_credential_list:
        if not isinstance(http_write_credential_list, list):
            msg = 'Global HTTP write credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('httpsWrite') is None:
            msg = 'Global HTTP write credentials are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(http_write_credential_list) > 0:
            global_credentials_all['httpsWrite'] = []
            cred_len = len(http_write_credential_list)
            if cred_len > 5:
                cred_len = 5
            for http_cred in http_write_credential_list:
                if http_cred.get('description') and http_cred.get('username'):
                    for http in response.get('httpsWrite'):
                        if http.get('description') == http.get('description') and http.get('username') == http.get('username'):
                            global_credentials_all['httpsWrite'].append(http.get('id'))
                    global_credentials_all['httpsWrite'] = global_credentials_all['httpsWrite'][:cred_len]
                else:
                    msg = 'Kindly ensure you include both the description and the username for the Global HTTP Write credential to discover the devices'
                    self.discovery_specific_cred_failure(msg=msg)
    snmp_v2_read_credential_list = global_credentials.get('snmp_v2_read_credential_list')
    if snmp_v2_read_credential_list:
        if not isinstance(snmp_v2_read_credential_list, list):
            msg = 'Global SNMPv2 read credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('snmpV2cRead') is None:
            msg = 'Global SNMPv2 read credentials are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(snmp_v2_read_credential_list) > 0:
            global_credentials_all['snmpV2cRead'] = []
            cred_len = len(snmp_v2_read_credential_list)
            if cred_len > 5:
                cred_len = 5
            for snmp_cred in snmp_v2_read_credential_list:
                if snmp_cred.get('description'):
                    for snmp in response.get('snmpV2cRead'):
                        if snmp.get('description') == snmp_cred.get('description'):
                            global_credentials_all['snmpV2cRead'].append(snmp.get('id'))
                    global_credentials_all['snmpV2cRead'] = global_credentials_all['snmpV2cRead'][:cred_len]
                else:
                    msg = 'Kindly ensure you include the description for the Global SNMPv2 Read                                 credential to discover the devices'
                    self.discovery_specific_cred_failure(msg=msg)
    snmp_v2_write_credential_list = global_credentials.get('snmp_v2_write_credential_list')
    if snmp_v2_write_credential_list:
        if not isinstance(snmp_v2_write_credential_list, list):
            msg = 'Global SNMPv2 write credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('snmpV2cWrite') is None:
            msg = 'Global SNMPv2 write credentials are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(snmp_v2_write_credential_list) > 0:
            global_credentials_all['snmpV2cWrite'] = []
            cred_len = len(snmp_v2_write_credential_list)
            if cred_len > 5:
                cred_len = 5
            for snmp_cred in snmp_v2_write_credential_list:
                if snmp_cred.get('description'):
                    for snmp in response.get('snmpV2cWrite'):
                        if snmp.get('description') == snmp_cred.get('description'):
                            global_credentials_all['snmpV2cWrite'].append(snmp.get('id'))
                    global_credentials_all['snmpV2cWrite'] = global_credentials_all['snmpV2cWrite'][:cred_len]
                else:
                    msg = 'Kindly ensure you include the description for the Global SNMPV2 write credential to discover the devices'
                    self.discovery_specific_cred_failure(msg=msg)
    snmp_v3_credential_list = global_credentials.get('snmp_v3_credential_list')
    if snmp_v3_credential_list:
        if not isinstance(snmp_v3_credential_list, list):
            msg = 'Global SNMPv3 write credentials must be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('snmpV3') is None:
            msg = 'Global SNMPv3 credentials are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(snmp_v3_credential_list) > 0:
            global_credentials_all['snmpV3'] = []
            cred_len = len(snmp_v3_credential_list)
            if cred_len > 5:
                cred_len = 5
            for snmp_cred in snmp_v3_credential_list:
                if snmp_cred.get('description') and snmp_cred.get('username'):
                    for snmp in response.get('snmpV3'):
                        if snmp.get('description') == snmp_cred.get('description') and snmp.get('username') == snmp_cred.get('username'):
                            global_credentials_all['snmpV3'].append(snmp.get('id'))
                    global_credentials_all['snmpV3'] = global_credentials_all['snmpV3'][:cred_len]
                else:
                    msg = 'Kindly ensure you include both the description and the username for the Global SNMPv3                                 to discover the devices'
                    self.discovery_specific_cred_failure(msg=msg)
    net_conf_port_list = global_credentials.get('net_conf_port_list')
    if net_conf_port_list:
        if not isinstance(net_conf_port_list, list):
            msg = 'Global net Conf Ports be passed as a list'
            self.discovery_specific_cred_failure(msg=msg)
        if response.get('netconfCredential') is None:
            msg = 'Global netconf ports are not present in the Cisco Catalyst Center'
            self.discovery_specific_cred_failure(msg=msg)
        if len(net_conf_port_list) > 0:
            global_credentials_all['netconfCredential'] = []
            cred_len = len(net_conf_port_list)
            if cred_len > 5:
                cred_len = 5
            for port in net_conf_port_list:
                if port.get('description'):
                    for netconf in response.get('netconfCredential'):
                        if port.get('description') == netconf.get('description'):
                            global_credentials_all['netconfCredential'].append(netconf.get('id'))
                    global_credentials_all['netconfCredential'] = global_credentials_all['netconfCredential'][:cred_len]
                else:
                    msg = 'Please provide valid description of the Global Netconf port to be used'
                    self.discovery_specific_cred_failure(msg=msg)
    self.log('Fetched Global credentials IDs are {0}'.format(global_credentials_all), 'INFO')
    return global_credentials_all