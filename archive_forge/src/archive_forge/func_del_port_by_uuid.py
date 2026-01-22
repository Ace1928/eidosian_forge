import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def del_port_by_uuid(manager, system_id, bridge_name, port_uuid):
    return del_port(manager, system_id, bridge_name, lambda tables: _get_port(tables, port_uuid, attr_name='uuid'))