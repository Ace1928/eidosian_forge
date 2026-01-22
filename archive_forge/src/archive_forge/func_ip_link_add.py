import logging
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import netdevice
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from . import base
@base.sql_function
def ip_link_add(session, name, type_='loopback', lladdr='00:00:00:00:00:00'):
    """
    Adds an interface record into Zebra protocol service database.

    The arguments are similar to "ip link add" command of iproute2.

    :param session: Session instance connecting to database.
    :param name: Name of interface.
    :param type_: Type of interface. 'loopback' or 'ethernet'.
    :param lladdr: Link layer address. Mostly MAC address.
    :return: Instance of added record or already existing record.
    """
    intf = ip_link_show(session, ifname=name)
    if intf:
        LOG.debug('Interface "%s" already exists: %s', intf.ifname, intf)
        return intf
    if type_ == 'ethernet':
        intf = Interface(ifname=name, flags=DEFAULT_ETH_FLAGS, ifmtu=DEFAULT_ETH_MTU, ifmtu6=DEFAULT_ETH_MTU, hw_addr=lladdr)
    else:
        intf = Interface(ifname=name, inet='127.0.0.1/8', inet6='::1/128')
    session.add(intf)
    return intf