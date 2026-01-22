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
def discover_schemas(connection):
    req = jsonrpc.Message.create_request('list_dbs', [])
    error, reply = transact_block(req, connection)
    if error or reply.error:
        return
    schemas = []
    for db in reply.result:
        if db != 'Open_vSwitch':
            continue
        req = jsonrpc.Message.create_request('get_schema', [db])
        error, reply = transact_block(req, connection)
        if error or reply.error:
            continue
        schemas.append(reply.result)
    return schemas