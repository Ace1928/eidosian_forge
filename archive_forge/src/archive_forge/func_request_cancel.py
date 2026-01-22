import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
@wraps(Layer1.request_cancel_workflow_execution)
def request_cancel(self):
    """RequestCancelWorkflowExecution."""
    return self._swf.request_cancel_workflow_execution(self.domain, self.workflowId, self.runId)