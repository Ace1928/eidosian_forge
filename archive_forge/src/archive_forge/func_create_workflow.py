from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def create_workflow(self, **attrs):
    """Create a new workflow from attributes

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.workflow.v2.workflow.Workflow`,
            comprised of the properties on the Workflow class.

        :returns: The results of workflow creation
        :rtype: :class:`~openstack.workflow.v2.workflow.Workflow`
        """
    return self._create(_workflow.Workflow, **attrs)