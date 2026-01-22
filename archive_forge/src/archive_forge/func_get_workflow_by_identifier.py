from keystoneauth1.exceptions import http as ka_exceptions
from mistralclient.api import base as mistral_base
from mistralclient.api import client as mistral_client
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_workflow_by_identifier(self, workflow_identifier):
    try:
        return self.client().workflows.get(workflow_identifier)
    except Exception as ex:
        if self.is_not_found(ex):
            raise exception.EntityNotFound(entity='Workflow', name=workflow_identifier)
        raise