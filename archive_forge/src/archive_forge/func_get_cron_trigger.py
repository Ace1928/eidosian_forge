from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def get_cron_trigger(self, cron_trigger):
    """Get a cron trigger

        :param cron_trigger: The value can be the name of a cron_trigger or
            :class:`~openstack.workflow.v2.cron_trigger.CronTrigger` instance.

        :returns: One :class:`~openstack.workflow.v2.cron_trigger.CronTrigger`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            cron triggers matching the criteria could be found.
        """
    return self._get(_cron_trigger.CronTrigger, cron_trigger)