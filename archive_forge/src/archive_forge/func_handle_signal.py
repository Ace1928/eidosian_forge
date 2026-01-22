from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.engine import translation
def handle_signal(self, details):
    data = details or self.properties.get(self.DEFAULT_EXECUTION_DATA)
    execution_args = {'job_id': self.resource_id, 'cluster_id': data.get(self.CLUSTER), 'input_id': data.get(self.INPUT), 'output_id': data.get(self.OUTPUT), 'is_public': data.get(self.IS_PUBLIC), 'interface': data.get(self.INTERFACE), 'configs': {'configs': data.get(self.CONFIGS), 'params': data.get(self.PARAMS), 'args': data.get(self.ARGS)}, 'is_protected': False}
    try:
        self.client().job_executions.create(**execution_args)
    except Exception as ex:
        raise exception.ResourceFailure(ex, self)