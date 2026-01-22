import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def get_ifaces_by_other_config(manager, system_id, key, value, fn=None):
    return rows_by_other_config(manager, system_id, key, value, 'Interface', fn)