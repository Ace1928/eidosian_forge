import contextlib
import functools
import logging
import os
import sys
import time
import traceback
from kazoo import client
from taskflow.conductors import backends as conductor_backends
from taskflow import engines
from taskflow.jobs import backends as job_backends
from taskflow import logging as taskflow_logging
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends as persistence_backends
from taskflow.persistence import models
from taskflow import task
from oslo_utils import timeutils
from oslo_utils import uuidutils
def run_poster():
    print('Starting poster with pid: %s' % ME)
    my_name = 'poster-%s' % ME
    persist_backend = persistence_backends.fetch(PERSISTENCE_URI)
    with contextlib.closing(persist_backend):
        with contextlib.closing(persist_backend.get_connection()) as conn:
            conn.upgrade()
        job_backend = job_backends.fetch(my_name, JB_CONF, persistence=persist_backend)
        job_backend.connect()
        with contextlib.closing(job_backend):
            lb = models.LogBook('post-from-%s' % my_name)
            fd = models.FlowDetail('song-from-%s' % my_name, uuidutils.generate_uuid())
            lb.add(fd)
            with contextlib.closing(persist_backend.get_connection()) as conn:
                conn.save_logbook(lb)
            engines.save_factory_details(fd, make_bottles, [HOW_MANY_BOTTLES], {}, backend=persist_backend)
            jb = job_backend.post('song-from-%s' % my_name, book=lb)
            print('Posted: %s' % jb)
            print('Goodbye...')