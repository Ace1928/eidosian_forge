from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from ansible.module_utils import six
def update_target_interface_info(self, retries=60):
    """Discover and update cached interface info."""
    net_interfaces = list()
    try:
        rc, net_interfaces = self.request('storage-systems/%s/configuration/ethernet-interfaces' % self.ssid)
    except Exception as error:
        if retries > 0:
            self.update_target_interface_info(retries=retries - 1)
            return
        else:
            self.module.fail_json(msg='Failed to retrieve defined management interfaces. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
    iface = None
    channels = {}
    controller_info = self.get_controllers()[self.controller]
    controller_ref = controller_info['controllerRef']
    controller_ssh = controller_info['ssh']
    controller_dns = None
    controller_ntp = None
    dummy_interface_id = None
    for net in net_interfaces:
        if net['controllerRef'] == controller_ref:
            channels.update({net['channel']: net['linkStatus']})
            if dummy_interface_id is None:
                dummy_interface_id = net['interfaceRef']
            if controller_dns is None:
                controller_dns = net['dnsProperties']
            if controller_ntp is None:
                controller_ntp = net['ntpProperties']
        if net['ipv4Enabled'] and net['linkStatus'] == 'up':
            self.all_interface_addresses.append(net['ipv4Address'])
        if net['controllerRef'] == controller_ref and net['channel'] == self.channel:
            iface = net
        elif net['ipv4Enabled'] and net['linkStatus'] == 'up':
            self.alt_interface_addresses.append(net['ipv4Address'])
    self.interface_info.update({'id': dummy_interface_id, 'controllerRef': controller_ref, 'ssh': controller_ssh, 'dns_config_method': controller_dns['acquisitionProperties']['dnsAcquisitionType'], 'dns_servers': controller_dns['acquisitionProperties']['dnsServers'], 'ntp_config_method': controller_ntp['acquisitionProperties']['ntpAcquisitionType'], 'ntp_servers': controller_ntp['acquisitionProperties']['ntpServers']})
    if self.config_method is not None:
        if iface is None:
            available_controllers = ['%s (%s)' % (channel, status) for channel, status in channels.items()]
            self.module.fail_json(msg='Invalid port number! Controller %s ports: [%s]. Array [%s]' % (self.controller, ','.join(available_controllers), self.ssid))
        else:
            self.interface_info.update({'id': iface['interfaceRef'], 'controllerSlot': iface['controllerSlot'], 'channel': iface['channel'], 'link_status': iface['linkStatus'], 'enabled': iface['ipv4Enabled'], 'config_method': iface['ipv4AddressConfigMethod'], 'address': iface['ipv4Address'], 'subnet_mask': iface['ipv4SubnetMask'], 'gateway': iface['ipv4GatewayAddress'], 'ipv6_enabled': iface['ipv6Enabled']})