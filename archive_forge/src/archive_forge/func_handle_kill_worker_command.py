import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def handle_kill_worker_command(worker: 'Worker', payload: Dict[Any, Any]):
    """
    Stops work horse

    Args:
        worker (Worker): The worker to stop
        payload (Dict[Any, Any]): The payload.
    """
    worker.log.info('Received kill horse command.')
    if worker.horse_pid:
        worker.log.info('Kiling horse...')
        worker.kill_horse()
    else:
        worker.log.info('Worker is not working, kill horse command ignored')