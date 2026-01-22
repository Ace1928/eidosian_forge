import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def execution_list(self, schedule, mistral_client=None, marker='', limit=None):
    """Get a list of all executions of a scheduled backup.

        :param: schedule for which to list executions.
        :rtype: list of :class:`ScheduleExecution`.
        """
    if isinstance(schedule, Schedule):
        schedule = schedule.id
    if isinstance(marker, ScheduleExecution):
        marker = getattr(marker, 'id')
    if not mistral_client:
        mistral_client = self._get_mistral_client()
    cron_trigger = mistral_client.cron_triggers.get(schedule)
    ct_input = json.loads(cron_trigger.workflow_input)

    def mistral_execution_generator():
        m = marker
        while True:
            try:
                the_list = mistral_client.executions.list(marker=m, limit=50, sort_dirs='desc')
                if the_list:
                    for the_item in the_list:
                        yield the_item
                    m = the_list[-1].id
                else:
                    return
            except StopIteration:
                return

    def execution_list_generator():
        yielded = 0
        for sexec in mistral_execution_generator():
            if sexec.workflow_name == cron_trigger.workflow_name and ct_input == json.loads(sexec.input):
                yield ScheduleExecution(self, sexec.to_dict(), loaded=True)
                yielded += 1
            if limit and yielded == limit:
                return
    return list(execution_list_generator())