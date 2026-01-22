from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def get_network_iface_ip(self, network_name):
    networks = [n for n in self._get_network_ifaces_by_name(network_name) if n.DriverDescription == self._HYPERV_VIRT_ADAPTER]
    if not networks:
        LOG.error('No vswitch was found with name: %s', network_name)
        return (None, None)
    ip_addr = self._scimv2.MSFT_NetIPAddress(InterfaceIndex=networks[0].InterfaceIndex, AddressFamily=self._IPV4_ADDRESS_FAMILY)
    if not ip_addr:
        LOG.error('No IP Address could be found for network: %s', network_name)
        return (None, None)
    return (ip_addr[0].IPAddress, ip_addr[0].PrefixLength)