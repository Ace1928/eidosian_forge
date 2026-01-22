from openstack import proxy
from openstack.workflow.v2 import cron_trigger as _cron_trigger
from openstack.workflow.v2 import execution as _execution
from openstack.workflow.v2 import workflow as _workflow
def find_cron_trigger(self, name_or_id, ignore_missing=True, *, all_projects=False, **query):
    """Find a single cron trigger

        :param name_or_id: The name or ID of a cron trigger.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the resource does not exist. When set to ``True``, None will be
            returned when attempting to find a nonexistent resource.
        :param bool all_projects: When set to ``True``, search for cron
            triggers by name across all projects. Note that this will likely
            result in a higher chance of duplicates.
        :param kwargs query: Optional query parameters to be sent to limit
            the cron triggers being returned.

        :returns: One :class:`~openstack.compute.v2.cron_trigger.CronTrigger`
            or None
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource can be found.
        :raises: :class:`~openstack.exceptions.DuplicateResource` when multiple
            resources are found.
        """
    return self._find(_cron_trigger.CronTrigger, name_or_id, ignore_missing=ignore_missing, all_projects=all_projects, **query)