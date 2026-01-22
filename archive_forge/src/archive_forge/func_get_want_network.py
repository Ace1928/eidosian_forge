from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_want_network(self, network_management_details):
    """
        Get all the Network related information from playbook
        Set the status and the msg before returning from the API
        Check the return value of the API with check_return_status()

        Parameters:
            network_management_details (dict) - Playbook network
            details containing various network settings.

        Returns:
            self - The current object with updated desired Network-related information.
        """
    want_network = {'settings': {'dhcpServer': {}, 'dnsServer': {}, 'snmpServer': {}, 'syslogServer': {}, 'netflowcollector': {}, 'ntpServer': {}, 'timezone': '', 'messageOfTheday': {}, 'network_aaa': {}, 'clientAndEndpoint_aaa': {}}}
    want_network_settings = want_network.get('settings')
    self.log('Current state (have): {0}'.format(self.have), 'DEBUG')
    if network_management_details.get('dhcp_server') is not None:
        want_network_settings.update({'dhcpServer': network_management_details.get('dhcp_server')})
    else:
        del want_network_settings['dhcpServer']
    if network_management_details.get('ntp_server') is not None:
        want_network_settings.update({'ntpServer': network_management_details.get('ntp_server')})
    else:
        del want_network_settings['ntpServer']
    if network_management_details.get('timezone') is not None:
        want_network_settings['timezone'] = network_management_details.get('timezone')
    else:
        self.msg = 'missing parameter timezone in network'
        self.status = 'failed'
        return self
    dnsServer = network_management_details.get('dns_server')
    if dnsServer is not None:
        if dnsServer.get('domain_name') is not None:
            want_network_settings.get('dnsServer').update({'domainName': dnsServer.get('domain_name')})
        if dnsServer.get('primary_ip_address') is not None:
            want_network_settings.get('dnsServer').update({'primaryIpAddress': dnsServer.get('primary_ip_address')})
        if dnsServer.get('secondary_ip_address') is not None:
            want_network_settings.get('dnsServer').update({'secondaryIpAddress': dnsServer.get('secondary_ip_address')})
    else:
        del want_network_settings['dnsServer']
    snmpServer = network_management_details.get('snmp_server')
    if snmpServer is not None:
        if snmpServer.get('configure_dnac_ip') is not None:
            want_network_settings.get('snmpServer').update({'configureDnacIP': snmpServer.get('configure_dnac_ip')})
        if snmpServer.get('ip_addresses') is not None:
            want_network_settings.get('snmpServer').update({'ipAddresses': snmpServer.get('ip_addresses')})
    else:
        del want_network_settings['snmpServer']
    syslogServer = network_management_details.get('syslog_server')
    if syslogServer is not None:
        if syslogServer.get('configure_dnac_ip') is not None:
            want_network_settings.get('syslogServer').update({'configureDnacIP': syslogServer.get('configure_dnac_ip')})
        if syslogServer.get('ip_addresses') is not None:
            want_network_settings.get('syslogServer').update({'ipAddresses': syslogServer.get('ip_addresses')})
    else:
        del want_network_settings['syslogServer']
    netflowcollector = network_management_details.get('netflow_collector')
    if netflowcollector is not None:
        if netflowcollector.get('ip_address') is not None:
            want_network_settings.get('netflowcollector').update({'ipAddress': netflowcollector.get('ip_address')})
        if netflowcollector.get('port') is not None:
            want_network_settings.get('netflowcollector').update({'port': netflowcollector.get('port')})
    else:
        del want_network_settings['netflowcollector']
    messageOfTheday = network_management_details.get('message_of_the_day')
    if messageOfTheday is not None:
        if messageOfTheday.get('banner_message') is not None:
            want_network_settings.get('messageOfTheday').update({'bannerMessage': messageOfTheday.get('banner_message')})
        if messageOfTheday.get('retain_existing_banner') is not None:
            want_network_settings.get('messageOfTheday').update({'retainExistingBanner': messageOfTheday.get('retain_existing_banner')})
    else:
        del want_network_settings['messageOfTheday']
    network_aaa = network_management_details.get('network_aaa')
    if network_aaa:
        if network_aaa.get('ip_address'):
            want_network_settings.get('network_aaa').update({'ipAddress': network_aaa.get('ip_address')})
        elif network_aaa.get('servers') == 'ISE':
            self.msg = 'missing parameter ip_address in network_aaa, server ISE is set'
            self.status = 'failed'
            return self
        if network_aaa.get('network'):
            want_network_settings.get('network_aaa').update({'network': network_aaa.get('network')})
        else:
            self.msg = 'missing parameter network in network_aaa'
            self.status = 'failed'
            return self
        if network_aaa.get('protocol'):
            want_network_settings.get('network_aaa').update({'protocol': network_aaa.get('protocol')})
        else:
            self.msg = 'missing parameter protocol in network_aaa'
            self.status = 'failed'
            return self
        if network_aaa.get('servers'):
            want_network_settings.get('network_aaa').update({'servers': network_aaa.get('servers')})
        else:
            self.msg = 'missing parameter servers in network_aaa'
            self.status = 'failed'
            return self
        if network_aaa.get('shared_secret'):
            want_network_settings.get('network_aaa').update({'sharedSecret': network_aaa.get('shared_secret')})
    else:
        del want_network_settings['network_aaa']
    clientAndEndpoint_aaa = network_management_details.get('client_and_endpoint_aaa')
    if clientAndEndpoint_aaa:
        if clientAndEndpoint_aaa.get('ip_address'):
            want_network_settings.get('clientAndEndpoint_aaa').update({'ipAddress': clientAndEndpoint_aaa.get('ip_address')})
        elif clientAndEndpoint_aaa.get('servers') == 'ISE':
            self.msg = 'missing parameter ip_address in clientAndEndpoint_aaa,                         server ISE is set'
            self.status = 'failed'
            return self
        if clientAndEndpoint_aaa.get('network'):
            want_network_settings.get('clientAndEndpoint_aaa').update({'network': clientAndEndpoint_aaa.get('network')})
        else:
            self.msg = 'missing parameter network in clientAndEndpoint_aaa'
            self.status = 'failed'
            return self
        if clientAndEndpoint_aaa.get('protocol'):
            want_network_settings.get('clientAndEndpoint_aaa').update({'protocol': clientAndEndpoint_aaa.get('protocol')})
        else:
            self.msg = 'missing parameter protocol in clientAndEndpoint_aaa'
            self.status = 'failed'
            return self
        if clientAndEndpoint_aaa.get('servers'):
            want_network_settings.get('clientAndEndpoint_aaa').update({'servers': clientAndEndpoint_aaa.get('servers')})
        else:
            self.msg = 'missing parameter servers in clientAndEndpoint_aaa'
            self.status = 'failed'
            return self
        if clientAndEndpoint_aaa.get('shared_secret'):
            want_network_settings.get('clientAndEndpoint_aaa').update({'sharedSecret': clientAndEndpoint_aaa.get('shared_secret')})
    else:
        del want_network_settings['clientAndEndpoint_aaa']
    self.log('Network playbook details: {0}'.format(want_network), 'DEBUG')
    self.want.update({'wantNetwork': want_network})
    self.msg = 'Collecting the network details from the playbook'
    self.status = 'success'
    return self