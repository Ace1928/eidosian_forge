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
def on_conductor_event(cond, event, details):
    print("Event '%s' has been received..." % event)
    print('Details = %s' % details)
    if event.endswith('_start'):
        w = timeutils.StopWatch()
        w.start()
        base_event = event[0:-len('_start')]
        event_watches[base_event] = w
    if event.endswith('_end'):
        base_event = event[0:-len('_end')]
        try:
            w = event_watches.pop(base_event)
            w.stop()
            print("It took %0.3f seconds for event '%s' to finish" % (w.elapsed(), base_event))
        except KeyError:
            pass
    if event == 'running_end' and only_run_once:
        cond.stop()