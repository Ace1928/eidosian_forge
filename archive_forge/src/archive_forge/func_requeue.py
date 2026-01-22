import logging
import logging.config
import os
import sys
import warnings
from typing import List, Type
import click
from redis.exceptions import ConnectionError
from rq import Connection, Retry
from rq import __version__ as version
from rq.cli.helpers import (
from rq.contrib.legacy import cleanup_ghosts
from rq.defaults import (
from rq.exceptions import InvalidJobOperationError
from rq.job import Job, JobStatus
from rq.logutils import blue
from rq.registry import FailedJobRegistry, clean_registries
from rq.serializers import DefaultSerializer
from rq.suspension import is_suspended
from rq.suspension import resume as connection_resume
from rq.suspension import suspend as connection_suspend
from rq.utils import get_call_string, import_attribute
from rq.worker import Worker
from rq.worker_pool import WorkerPool
from rq.worker_registration import clean_worker_registry
@main.command()
@click.option('--all', '-a', is_flag=True, help='Requeue all failed jobs')
@click.option('--queue', required=True, type=str)
@click.argument('job_ids', nargs=-1)
@pass_cli_config
def requeue(cli_config, queue, all, job_class, serializer, job_ids, **options):
    """Requeue failed jobs."""
    failed_job_registry = FailedJobRegistry(queue, connection=cli_config.connection, job_class=job_class, serializer=serializer)
    if all:
        job_ids = failed_job_registry.get_job_ids()
    if not job_ids:
        click.echo('Nothing to do')
        sys.exit(0)
    click.echo('Requeueing {0} jobs from failed queue'.format(len(job_ids)))
    fail_count = 0
    with click.progressbar(job_ids) as job_ids:
        for job_id in job_ids:
            try:
                failed_job_registry.requeue(job_id)
            except InvalidJobOperationError:
                fail_count += 1
    if fail_count > 0:
        click.secho('Unable to requeue {0} jobs from failed job registry'.format(fail_count), fg='red')