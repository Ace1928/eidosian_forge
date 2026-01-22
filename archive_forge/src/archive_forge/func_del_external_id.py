import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def del_external_id(manager, system_id, key, fn):

    def _del_iface_external_id(tables, *_):
        row = fn(tables)
        if not row:
            return None
        external_ids = row.external_ids
        if key in external_ids:
            external_ids.pop(key)
            row.external_ids = external_ids
    req = ovsdb_event.EventModifyRequest(system_id, _del_iface_external_id)
    return manager.send_request(req)