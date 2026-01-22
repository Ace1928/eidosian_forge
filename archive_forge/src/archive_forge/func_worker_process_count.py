import os
from oslo_service import service
import setproctitle
from neutron_lib.callbacks import events
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
@property
def worker_process_count(self):
    """The worker's process count.

        :returns: The number of processes to spawn for this worker.
        """
    return self._worker_process_count