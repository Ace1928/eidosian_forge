import logging
import socket
from sqlalchemy import Column
from sqlalchemy import Boolean
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import ip
from os_ken.lib.packet import safi as packet_safi
from os_ken.lib.packet import zebra
from . import base
from . import interface
@base.sql_function
def ip_route_show(session, destination, device, **kwargs):
    """
    Returns a selected route record matching the given filtering rules.

    The arguments are similar to "ip route showdump" command of iproute2.

    :param session: Session instance connecting to database.
    :param destination: Destination prefix.
    :param device: Source device.
    :param kwargs: Filtering rules to query.
    :return: Instance of route record or "None" if failed.
    """
    intf = interface.ip_link_show(session, ifname=device)
    if not intf:
        LOG.debug('Interface "%s" does not exist', device)
        return None
    return session.query(Route).filter_by(destination=destination, ifindex=intf.ifindex, **kwargs).first()