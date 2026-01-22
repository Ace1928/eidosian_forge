import asyncio
import logging
import time
import traceback
from .compatibility import guarantee_single_callable
def delete_oldest_application_instance(self):
    """
        Finds and deletes the oldest application instance
        """
    oldest_time = min((details['last_used'] for details in self.application_instances.values()))
    for scope_id, details in self.application_instances.items():
        if details['last_used'] == oldest_time:
            self.delete_application_instance(scope_id)
            return