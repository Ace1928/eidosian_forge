import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
def discover_system_id(idl):
    system_id = None
    while system_id is None and idl._session.is_connected():
        idl.run()
        openvswitch = idl.tables['Open_vSwitch'].rows
        if openvswitch:
            row = openvswitch.get(list(openvswitch.keys())[0])
            system_id = row.external_ids.get('system-id')
    return system_id