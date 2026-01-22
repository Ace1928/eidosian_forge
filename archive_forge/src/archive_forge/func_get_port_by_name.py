import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_port_by_name(manager, system_id, name, fn=None):
    port = row_by_name(manager, system_id, name, 'Port')
    if fn is not None:
        return fn(port)
    return port