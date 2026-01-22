import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def handle_shutdown_command(worker: 'Worker'):
    """Perform shutdown command.

    Args:
        worker (Worker): The worker to use.
    """
    worker.log.info('Received shutdown command, sending SIGINT signal.')
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)