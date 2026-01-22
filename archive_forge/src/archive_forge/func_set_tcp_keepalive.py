import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def set_tcp_keepalive(sock, tcp_keepalive=True, tcp_keepidle=None, tcp_keepalive_interval=None, tcp_keepalive_count=None):
    """Set values for tcp keepalive parameters

    This function configures tcp keepalive parameters if users wish to do
    so.

    :param tcp_keepalive: Boolean, turn on or off tcp_keepalive. If users are
      not sure, this should be True, and default values will be used.

    :param tcp_keepidle: time to wait before starting to send keepalive probes
    :param tcp_keepalive_interval: time between successive probes, once the
      initial wait time is over
    :param tcp_keepalive_count: number of probes to send before the connection
      is killed
    """
    if isinstance(tcp_keepalive, bool):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, tcp_keepalive)
    else:
        raise TypeError('tcp_keepalive must be a boolean')
    if not tcp_keepalive:
        return
    if tcp_keepidle is not None:
        if hasattr(socket, 'TCP_KEEPIDLE'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, tcp_keepidle)
        else:
            LOG.warning('tcp_keepidle not available on your system')
    if tcp_keepalive_interval is not None:
        if hasattr(socket, 'TCP_KEEPINTVL'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, tcp_keepalive_interval)
        else:
            LOG.warning('tcp_keepintvl not available on your system')
    if tcp_keepalive_count is not None:
        if hasattr(socket, 'TCP_KEEPCNT'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, tcp_keepalive_count)
        else:
            LOG.warning('tcp_keepcnt not available on your system')