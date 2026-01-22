import os
import socket
import struct
from os_ken import cfg
from os_ken.base.app_manager import OSKenApp
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.lib.packet import safi as packet_safi
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client import event as zclient_event
def send_ip_route_delete(self, prefix, nexthops=None, safi=packet_safi.UNICAST, flags=zebra.ZEBRA_FLAG_INTERNAL, distance=None, metric=None, mtu=None, tag=None):
    """
        Sends ZEBRA_IPV4/v6_ROUTE_DELETE message to Zebra daemon.

        :param prefix: IPv4/v6 Prefix to advertise.
        :param nexthops: List of nexthop addresses.
        :param safi: SAFI to advertise.
        :param flags: Message flags to advertise. See "ZEBRA_FLAG_*".
        :param distance: (Optional) Distance to advertise.
        :param metric: (Optional) Metric to advertise.
        :param mtu: (Optional) MTU size to advertise.
        :param tag: (Optional) TAG information to advertise.
        :return: Zebra message instance to be sent. None if failed.
        """
    try:
        return self._send_ip_route_impl(prefix=prefix, nexthops=nexthops, safi=safi, flags=flags, distance=distance, metric=metric, mtu=mtu, tag=tag, is_withdraw=True)
    except ValueError as e:
        self.logger.exception('Cannot send IP route delete message: %s', e)
        return None