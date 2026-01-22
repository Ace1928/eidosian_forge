import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size
import click
from redis import Redis
from redis.sentinel import Sentinel
from rq.defaults import (
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus
def show_workers(queues, raw, by_queue, queue_class, worker_class):
    workers = set()
    if queues:
        for queue in queues:
            for worker in worker_class.all(queue=queue):
                workers.add(worker)
    else:
        for worker in worker_class.all():
            workers.add(worker)
    if not by_queue:
        for worker in workers:
            queue_names = ', '.join(worker.queue_names())
            name = '%s (%s %s %s)' % (worker.name, worker.hostname, worker.ip_address, worker.pid)
            if not raw:
                line = '%s: %s %s. jobs: %d finished, %d failed' % (name, state_symbol(worker.get_state()), queue_names, worker.successful_job_count, worker.failed_job_count)
                click.echo(line)
            else:
                line = 'worker %s %s %s. jobs: %d finished, %d failed' % (name, worker.get_state(), queue_names, worker.successful_job_count, worker.failed_job_count)
                click.echo(line)
    else:
        queue_dict = {}
        for queue in queues:
            queue_dict[queue] = worker_class.all(queue=queue)
        if queue_dict:
            max_length = max((len(q.name) for q, in queue_dict.keys()))
        else:
            max_length = 0
        for queue in queue_dict:
            if queue_dict[queue]:
                queues_str = ', '.join(sorted(map(lambda w: '%s (%s)' % (w.name, state_symbol(w.get_state())), queue_dict[queue])))
            else:
                queues_str = '–'
            click.echo('%s %s' % (pad(queue.name + ':', max_length + 1), queues_str))
    if not raw:
        click.echo('%d workers, %d queues' % (len(workers), len(queues)))