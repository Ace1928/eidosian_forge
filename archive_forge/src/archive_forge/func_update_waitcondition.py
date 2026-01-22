from heat.api.aws import exception
from heat.common import identifier
from heat.common import wsgi
from heat.rpc import client as rpc_client
def update_waitcondition(self, req, body, arn):
    con = req.context
    identity = identifier.ResourceIdentifier.from_arn(arn)
    try:
        md = self.rpc_client.resource_signal(con, stack_identity=dict(identity.stack()), resource_name=identity.resource_name, details=body, sync_call=True)
    except Exception as ex:
        return exception.map_remote_error(ex)
    return {'resource': identity.resource_name, 'metadata': md}