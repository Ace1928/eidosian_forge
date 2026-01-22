import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def move_task_by_id(task_id, dest, **kwargs):
    """Find a task by id and move it to another queue.

    Arguments:
        task_id (str): Id of task to find and move.
        dest: (str, kombu.Queue): Destination queue.
        transform (Callable): Optional function to transform the return
            value (destination) of the filter function.
        **kwargs (Any): Also supports the same keyword
            arguments as :func:`move`.
    """
    return move_by_idmap({task_id: dest}, **kwargs)