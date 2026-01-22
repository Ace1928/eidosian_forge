import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def send_shutdown_command(connection: 'Redis', worker_name: str):
    """
    Sends a command to shutdown a worker.

    Args:
        connection (Redis): A Redis Connection
        worker_name (str): The Job ID
    """
    send_command(connection, worker_name, 'shutdown')