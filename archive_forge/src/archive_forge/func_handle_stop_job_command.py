import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def handle_stop_job_command(worker: 'Worker', payload: Dict[Any, Any]):
    """Handles stop job command.

    Args:
        worker (Worker): The worker to use
        payload (Dict[Any, Any]): The payload.
    """
    job_id = payload.get('job_id')
    worker.log.debug('Received command to stop job %s', job_id)
    if job_id and worker.get_current_job_id() == job_id:
        worker._stopped_job_id = job_id
        worker.kill_horse()
    else:
        worker.log.info('Not working on job %s, command ignored.', job_id)