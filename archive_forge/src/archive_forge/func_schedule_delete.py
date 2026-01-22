import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def schedule_delete(self, schedule, mistral_client=None):
    """Remove a given backup schedule.

        :param schedule: schedule to delete.
        """
    if isinstance(schedule, Schedule):
        schedule = schedule.id
    if not mistral_client:
        mistral_client = self._get_mistral_client()
    mistral_client.cron_triggers.delete(schedule)