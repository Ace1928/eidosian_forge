import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def del_iface_external_id(manager, system_id, iface_name, key):
    return del_external_id(manager, system_id, key, lambda tables: _get_iface(tables, iface_name))