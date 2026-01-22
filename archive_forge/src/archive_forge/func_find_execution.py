from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def find_execution(self, name_or_id, ignore_missing=True):
    """Find a single execution

        :param name_or_id: The name or ID of an execution.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.
        :returns: One :class:`~openstack.compute.v2.execution.Execution` or
            None
        """
    return self._find(_execution.Execution, name_or_id, ignore_missing=ignore_missing)